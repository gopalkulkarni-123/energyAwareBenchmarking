# Decode-phase DVFS and power-cap benchmarking

Tools for measuring how GPU power caps and SM clock locking affect the energy
and latency of LLM decode, per token.

Built to test a claim from Ma et al., *The Illusion of Power Capping in LLM
Decode*: that power capping is structurally ineffective for memory-bound
decode. On A100 80GB PCIe it is not — see `rho_eta_criterion.md` for the
working theory of when it holds and when it does not.

---

## Files

| file | role |
|---|---|
| `dvfs_probe.py` | sets the knobs, samples NVML, sweeps cap and clock |
| `bench_vllm.py` | the workload: vLLM decode with phase-separated timing |
| `decode_pareto/extract.py` | sweep CSVs → per-configuration decode metrics |
| `decode_pareto/pareto.py` | metrics → figures and Pareto front |
| `rho_eta_criterion.md` | theory notes: when does a power cap actually engage |

```
dvfs_probe.py sweep --out X.csv      ← run this
        ↓  calls as a subprocess
bench_vllm.py                        ← the workload
        ↓  X.csv
decode_pareto/extract.py             ← decode_metrics.csv
        ↓
decode_pareto/pareto.py              ← figures + pareto_front.csv
```

The two halves are coupled only through three agreed-upon stdout lines, so any
workload that emits them can replace `bench_vllm.py`:

```
MARK <name> <epoch>       measurement window boundary; <x>_start / <x>_end pairs
WORKLOAD_METRIC <float>   throughput proxy (decode tokens)
WORKLOAD_JSON   {...}     extra fields, merged into every CSV row
```

---

## One-time setup

```bash
pip install nvidia-ml-py            # dvfs_probe
pip install matplotlib             # analysis only
```

`dvfs_probe.py` runs as your **normal user**. Only the three state-changing
`nvidia-smi` calls are elevated, through `sudo -n`. Grant that once:

```bash
which nvidia-smi                    # usually /usr/bin/nvidia-smi
sudo visudo -f /etc/sudoers.d/nvidia-smi
```

Add one line, substituting your username and the path above:

```
<user> ALL=(root) NOPASSWD: /usr/bin/nvidia-smi
```

Verify — this must print the GPU list with no password prompt:

```bash
sudo -n nvidia-smi -L
```

Do **not** run `sudo python dvfs_probe.py`. The workload would then execute as
root, lose your environment (venv, CUDA toolkit, HuggingFace cache), and can
leave root-owned files in `~/.cache` that break later runs.

Enable persistence mode. Without it the driver tears down GPU context whenever
no client is attached, so clock and power settings can silently revert between
runs, and idle readings are meaningless:

```bash
sudo nvidia-smi -pm 1
```

---

## Step 1: probe the hardware

Read-only, a few seconds, no privileges needed:

```bash
python dvfs_probe.py probe --gpu 0
```

Three numbers matter:

- **default limit** — the TDP
- **settable range** — the lower bound is `cap_min`; `ρ = cap_min / TDP`
- **supported MEM steps** — a single step means `-lmc` is a no-op on this card
  (true of every HBM part so far)

The Privileges section confirms the sudoers rule. Add `--test-clamp` to check
whether the card honours a requested clock lock exactly; H200 silently clamps
1980 MHz to 1830, A100 does not.

---

## Step 2: validate the workload before touching the GPU

```bash
python dvfs_probe.py sweep --gpu 0 --preflight-only \
  --workload "env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    python bench_vllm.py --model ./qwen3_4b/ --bs 1 --in-len 1024 --out-len 512 \
    --method double-len --warmup 1"
```

This runs the workload once with its output streamed to the terminal, changes
no knob, and exits with the workload's exit code. Use it whenever a workload
command is new or failing.

Success looks like four `MARK` lines, one `WORKLOAD_JSON`, one
`WORKLOAD_METRIC`, and `exit code 0`.

Pin the GPU explicitly with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES`. NVML orders devices by PCI bus; CUDA defaults to
fastest-first. On a multi-GPU node they can disagree, and you would then set
the clock on one card while running the workload on another.

---

## Step 3: sweep

```bash
python dvfs_probe.py sweep --gpu 0 --reps 5 \
  --caps 150,225,300 --clocks 510,810,1110,1410 \
  --workload "env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    python bench_vllm.py --model ./qwen3_4b/ --bs 1 --in-len 1024 --out-len 512 \
    --method double-len --warmup 1" \
  --out bs1_seq1024.csv
```

Order of operations: privilege check, then a preflight run at stock settings,
then the cap sweep with the clock unlocked, then the clock sweep with the cap
restored to default. Knobs are reset in a `finally` block and again at exit.

Omit `--caps` / `--clocks` for five evenly spaced points across the hardware
range. Drop 210 MHz from any real sweep: it is dominated everywhere and 3–6×
slower than anything else, so it only costs time.

### Reading the output

```
[cap  ]    150       full |  12.35 s | P_med 150.3 W | SM_med  525 MHz | E 1841.1 J | lim   150W | cap  99.2%
[clock]    510       full |  12.54 s | P_med 147.5 W | SM_med  510 MHz | E 1842.1 J | lim   300W | cap   0.0%
```

- `lim` — the power limit **actually in force**. The clock sweep restores it to
  default, so a 0% there and a 99% in the cap sweep are not contradictory: they
  face different walls. Always read `cap %` together with `lim`.
- `cap %` — fraction of samples where the GPU reported `SwPowerCap`, i.e. "the
  reason my clock is not higher is the power limit". It is an attribution of
  which constraint binds, **not** a claim that power exceeded the cap. A cap
  regulating correctly sits *at* its setpoint with a high `cap %`.
- `E` — trapezoidal integral of the 50 ms power samples over that window.
  **Not comparable across settings**: the workload auto-calibrates its
  repetition count, so different settings can do different amounts of work.
  Always normalise per token.
- `phase` — `prefill` and `full` are the two measurement windows (see below);
  `whole` covers the entire subprocess including model load, and is only for
  reconciliation.

### Long runs

Use tmux, not `nohup`. If `dvfs_probe.py` is killed with `SIGKILL` the GPU stays
modified; `SIGINT` runs the reset path.

```bash
tmux new -s sweep
./seq_sweep.sh 2>&1 | tee sweep.log
# Ctrl-B then D to detach; tmux attach -t sweep to return
```

After any sweep, confirm the card is clean:

```bash
nvidia-smi --query-gpu=index,clocks.sm,power.limit,power.draw --format=csv
sudo nvidia-smi -i 0 -rgc && sudo nvidia-smi -i 0 -pl 300   # if needed
```

Query once. Do not use `watch`: with persistence mode off, repeated polling
keeps the card pinned awake at max clock and roughly doubles its idle reading.

---

## How decode is isolated

Decode cannot be timed directly through the vLLM API, so `bench_vllm.py` times
two windows and subtracts. Two estimators are available.

### `--method double-len` (recommended)

Times `max_tokens=N` against `max_tokens=2N`, with the **same repetition count**
for both. Each window contains exactly one identically shaped prefill, so
prefill cancels exactly:

```
E_decode = E_window2 - E_window1        (prefill_scale = 1.0)
n_tokens = bs * out_len * reps
```

The difference is decode tokens N+1 … 2N, at a KV depth of `in_len + N` to
`in_len + 2N`. Both bounds are reported as `kv_depth_lo` / `kv_depth_hi`.

No scaling factor, no assumption that prefill's thermal state matches across
windows, and chunked prefill cancels too.

### `--method prefill-sub` (default, legacy)

Times a prefill-only window (`max_tokens=1`) and subtracts it, scaled by the
repetition ratio. Fine when prefill is a small fraction of the full window —
about 1% at BS=1, seq=1024. It degrades as that fraction grows: 19% at BS=32,
around 50% at seq=32K, where prefill measurement error propagates in proportion.

The two agree to 2.2% at BS=1, seq=1024, in the direction predicted by the
deeper KV depth that `double-len` measures.

### Window sizing

Repetition counts are auto-calibrated to reach `--target-window-s` (default 8 s,
≈160 samples at 50 ms). Two consequences:

- `window_s` is **not comparable across settings**. `n = int(target/dt)+1`
  flips between 1 and 2 right at `dt == target`, so two settings straddling the
  threshold get windows differing by 2× for no physical reason. Compare
  `decode_tok_per_s` or energy per token instead.
- For a sweep where equal work per setting matters, pin `--prefill-reps` and
  `--decode-reps` explicitly.

---

## Batch sweep

Batch size raises arithmetic intensity, `I ≈ 2NB / (2N + B·L·c_kv)`, while
weight traffic per step stays constant. This is the knob that moves decode
toward compute-bound, if anything does.

```bash
for B in 1 4 8 16 32; do
  python dvfs_probe.py sweep --gpu 0 --reps 5 \
    --caps 150,225,300 --clocks 510,810,1110,1410 \
    --workload "env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      python bench_vllm.py --model ./qwen3_4b/ --bs $B --in-len 1024 --out-len 512 \
      --method double-len --warmup 1" \
    --out bs${B}_seq1024.csv
done
```

Keep `--in-len` and `--out-len` fixed so batch is the only variable.

Measured so far on A100 80GB PCIe with Qwen3-4B: the energy-optimal clock stays
flat at 810 MHz from BS=1 to BS=32, because `I` only reaches 20 against a ridge
point of 161 FLOP/byte. Decode never gets close to compute-bound. This
reproduces the paper's *batch-invariant* class for GQA — on a card where the
cap does engage, which their framework does not anticipate.

Watch for `cap %` becoming nonzero at 1110 MHz as batch grows. At BS=32 it
first appears (0.6%), the only sign in that sweep of a qualitative change. BS=64
and 128 are worth adding for that reason.

---

## Sequence-length sweep

Context length enters through KV traffic, `B·L·c_kv`, which competes with the
`2N` weight traffic. The crossover where they match:

```
L_cross = 2N / (B · c_kv) ≈ 54,400 / B     for Qwen3-4B
```

Prediction: as `L` grows the decode step spends more time streaming KV with
less on-chip activity, so power falls and the cap may stop engaging. If A100
PCIe returns to the paper's regime under a workload parameter alone, that is
the sharpest evidence that cap effectiveness is not a hardware property.

### Feasible (B, L) region

KV cache is capped by what vLLM reports at startup:

```
GPU KV cache size: 458,208 tokens      ← Qwen3-4B at gpu_memory_utilization 0.9
```

The constraint is `B × (in_len + k·out_len) ≤ that number`, with `k = 2` under
`double-len`. Read the line from your own log; it shrinks as the model grows.

| L \ B | 1 | 4 | 8 | 16 | 32 | 128 |
|---|---|---|---|---|---|---|
| 1K | ok | ok | ok | ok | ok | ok |
| 4K | ok | ok | ok | ok | ok | — |
| 16K | ok | ok | ok | ok | — | — |
| 32K | ok | ok | ok | — | — | — |

BS=32 at 16K does **not** fit on this card with this model, even though the
paper has that cell. Qwen3-4B's native context is 32K; beyond that needs rope
scaling, which introduces another variable.

### Running it

`--out-len` must grow with `L`, or the decode window shrinks relative to an
increasingly expensive prefill (prefill attention is O(L²), decode is O(L)):

```bash
for L in 1024 4096 16384 32768; do
  OUT=$((L/4)); [ $OUT -lt 512 ] && OUT=512
  python dvfs_probe.py sweep --gpu 0 --reps 5 \
    --caps 150,225,300 --clocks 510,810,1110,1410 \
    --workload "env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      python bench_vllm.py --model ./qwen3_4b/ --bs 1 --in-len $L --out-len $OUT \
      --method double-len --warmup 1" \
    --out seq${L}_bs1.csv
done
```

Use `double-len` here specifically. Above `bs × in_len > 131072` the tool stops
forcing chunked prefill off, so prefill numbers include chunking — but under
`double-len` that cancels along with the rest of prefill.

---

## Analysis

```bash
cd decode_pareto
python extract.py          # reads sweep CSVs -> decode_metrics.csv
python pareto.py           # -> fig1..fig4 + pareto_front.csv
```

`extract.py` keys on the `model` and `bs` columns **inside** each file, not on
filenames, because an 8B run can easily end up with a name that looks like a 4B
batch-1 run.

`decode_w` is reconstructed as `E_decode / t_decode`, not read from
`power_median_w`, which still contains the prefill slice.

The Pareto front uses a **3% tolerance band**. Repeat-to-repeat spread of
J/token reaches 3.3% here (median 0.7%) on 2–3 repetitions, so a strict
dominance test ranks configurations that are statistically indistinguishable.
With the band, the front collapses into tie pairs: one cap point and one clock
point at the same physical operating point. That is the expected result once a
cap engages, since lowering the SM clock is the only actuator the driver has.

Raise `--reps` to 10 for anything you intend to publish.

---

## Known pitfalls

| symptom | cause |
|---|---|
| idle power roughly doubles, clock pinned at max | `watch nvidia-smi` polling with persistence mode off |
| workload fails only under `sudo` | do not run the tool under sudo; the workload loses PATH, CUDA toolkit and caches |
| `cap %` is 99% in one sweep and 0% in another at the same clock | different power limits in force; check the `lim` column |
| a slower setting appears to use less energy | different `decode_reps`; normalise per token |
| GPU left capped after a crash | `SIGKILL` skips the reset path; use `sudo nvidia-smi -rgc` and `-pl <TDP>` |
| FlashInfer JIT fails on `ninja` or an old `nvcc` | the CUDA 12+ toolkit is not on PATH, or `VLLM_USE_FLASHINFER_SAMPLER=0` to sidestep it |
