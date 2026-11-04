# The (ρ, η) criterion for power-cap effectiveness

Working notes. Status: **one hardware data point, one supporting data point.**
The criterion below is a hypothesis with a mechanism, not an established
result. Everything marked *predicted* is untested.

---

## 1. The problem it addresses

Ma et al., *The Illusion of Power Capping in LLM Decode* (arXiv 2605.11999),
argue that GPU power capping is **structurally** ineffective for autoregressive
decode, and that this result is scale-invariant:

> "This result is scale-invariant: decode arithmetic intensity is an
> algorithmic property unchanged by batch size, model size, or tensor
> parallelism."

Their evidence: on H200 (700 W TDP), decode draws 137–300 W, no cap from 280 W
to 700 W ever engages, and SM clock locking Pareto-dominates power capping
universally.

Our A100 80GB PCIe measurements contradict the generalisation while confirming
the underlying physics. Decode **is** memory-bound on both cards. What differs
is whether the power cap can reach the operating point at all.

The claim of this document: **cap effectiveness is not a property of the
workload, nor of the hardware, but of the (hardware, workload) pair** — and it
is predicted by two dimensionless ratios.

---

## 2. Definitions

| symbol | definition | source |
|---|---|---|
| `P_TDP` | default power limit | `nvmlDeviceGetPowerManagementDefaultLimit` |
| `cap_min` | lowest settable power limit | `nvmlDeviceGetPowerManagementLimitConstraints` |
| `P_decode` | decode-phase power at unconstrained clock | measured |
| **`ρ`** | `cap_min / P_TDP` — **actuator reach** | hardware only |
| **`η`** | `P_decode / P_TDP` — **power balance** | hardware × workload |

`ρ` is a hardware constant. `η` depends on the workload: model size, batch
size, context length, quantisation, and the software stack all move it.

### The criterion

```
ρ < η   ⟹  a settable cap exists that engages
ρ ≥ η   ⟹  the cap is inert at every settable value
```

Trivially true by construction: the cap engages iff some settable limit sits
below the natural draw. The content is not the inequality but the claim that
**these two numbers are the right coordinates**, that `η` is predictable from
workload parameters, and that the regime boundary explains published
disagreements.

### Regimes

| regime | condition | expected behaviour |
|---|---|---|
| **I. Inert** | `ρ ≥ η` | Cap never engages. Clock locking is the only lever. The paper's H200 case. |
| **II. Equivalent** | `ρ < η`, comfortably | Cap engages by lowering SM clock, converging on the same DVFS point as an explicit lock. Neither dominates. The A100 PCIe case. |
| **III. Binding** | `η → 1` | Cap engages even at settings an operator would realistically choose. *Predicted*, untested. |

Regime II matters most: it is where "clock locking Pareto-dominates" becomes
false, not because capping got better but because both knobs drive the same
actuator.

---

## 3. Measured points

### A100 80GB PCIe, Qwen3-4B BF16, vLLM 0.26, seq=1024

```
P_TDP    = 300 W
cap_min  = 150 W          ρ = 0.50
idle     ≈  48 W
```

| BS | `P_decode` | η | cap engaged? |
|---|---|---|---|
| 1 | 299 W | 1.00 | yes, 99% at every setting |
| 8 | 299 W | 1.00 | yes |
| 32 | 299 W | 1.00 | yes |

`ρ = 0.50 ≪ η = 1.00`: regime II, deeply.

Decode is nonetheless memory-bound throughout. At BS=1 the achieved weight
bandwidth is 1104 GB/s against a 1935 GB/s peak (57%), and tensor-core
utilisation is 0.35%. **The workload has not changed character; the card simply
has less power headroom relative to its memory system.**

### H200 SXM (from the paper)

```
P_TDP    = 700 W
cap_min  = 200 W          ρ = 0.29
P_decode = 137–300 W      η = 0.20–0.43
```

`ρ = 0.29` sits inside the `η` range, so the boundary is marginal. The paper
tested down to 280 W (0.40), above most of the `η` range, which is why nothing
engaged. **Part of their "no cap ever triggers" is a consequence of the sweep
range, not of the hardware floor at 200 W.** Worth stating carefully: at a
200 W cap, GQA at 207 W and MLA at 231 W should engage.

### Pareto consequence (A100 PCIe, measured)

With a 3% tolerance band (repeat-to-repeat spread reaches 3.3%), the front at
every batch size collapses into **tie pairs**, one cap point and one clock
point at the same physical operating point:

| BS | cap point | clock point |
|---|---|---|
| 8 | cap 225 → SM 1155 (1.091 ms, 0.244 J) | clock 1110 (1.099 ms, 0.239 J) |
| 16 | cap 225 → SM 1080 (0.634 ms, 0.142 J) | clock 1110 (0.625 ms, 0.144 J) |
| 32 | cap 225 → SM 1080 (0.429 ms, 0.097 J) | clock 1110 (0.423 ms, 0.098 J) |

Mechanism: when a cap engages, the driver's **only** actuator is SM
voltage/frequency. Both knobs therefore converge on the same DVFS point.
Capping is a closed-loop interface to the lever that clock locking drives
open-loop.

---

## 4. Why η is not a constant

`η` is the part that carries the physics. It rises with anything that
increases HBM traffic rate or on-chip activity.

### Independent of model size (measured)

BS=1 decode is weight-streaming bound: `Q ≈ 2N` bytes and `W ≈ 2N` FLOPs per
token, so arithmetic intensity `I ≈ 1` regardless of `N`. A larger model does
not raise the traffic *rate* unless bandwidth was not yet saturated.

Measured, 4B vs 8B at BS=1:

```
P_decode    299 W  →  299 W      unchanged
cap %        99%   →   99%       unchanged
BW achieved 1104   → 1434 GB/s   +30%   (bandwidth was NOT saturated at 4B)
mJ/token    2172   → 3417        +57%   = 2.05 (N) / 1.30 (BW)
```

`η` did not move. But the extra bandwidth **did** steal SM headroom: at a fixed
300 W cap the achieved SM clock fell from 1365 to 1290 MHz.

### The memory path costs ≈ 85 pJ/byte (measured)

Isolating at fixed locked clock, where the only difference between 4B and 8B is
traffic:

| clock | ΔBW | ΔP | implied |
|---|---|---|---|
| 510 MHz | +231 GB/s | +19.3 W | 84 pJ/byte |
| 810 MHz | +299 GB/s | +24.4 W | 82 pJ/byte |
| 1110 MHz | +363 GB/s | +31.9 W | 88 pJ/byte |

This covers DRAM, PHY, L2, interconnect and SM load/store units, not HBM alone
— which is why it exceeds the 4–7 pJ/bit device-level figures. It is an
empirical constant of this card, usable in an `η` model.

### Batch size: raises I, not η (measured)

```
I = 2NB / (2N + B·L·c_kv)
```

At `L=1024`, `I` goes 1 → 20 as `B` goes 1 → 32. But A100's ridge point is
`312/1.935 ≈ 161` FLOP/byte, so decode never approaches compute-bound, and the
energy-optimal clock stays flat at 810 MHz across all batch sizes. `η` stays
at 1.00.

This reproduces the paper's *batch-invariant* class for GQA — on a card where
the cap engages, which the paper's framework does not anticipate.

### Context length: predicted to lower η

`c_kv ≈ 147 KB/token` for Qwen3-4B (36 layers × 8 KV heads × 128 × 2 × 2 bytes;
verify against `config.json`). The crossover where KV traffic matches weight
traffic:

```
L_cross = 2N / (B · c_kv) = 54,400 / B
```

At long context the decode step spends more time streaming KV with lower
on-chip activity. **Prediction: `η` falls, and A100 PCIe may re-enter regime I.**
If so, the same card changes regime under a workload parameter alone — the
sharpest possible demonstration that cap effectiveness is not a hardware
property.

Untested. This is the single most valuable experiment remaining.

### Predicted movers of η

| knob | effect on η | reasoning | status |
|---|---|---|---|
| model size | none | traffic rate unchanged once BW saturates | measured |
| batch size | none (GQA) | `I` rises but stays far below ridge | measured |
| context length | ↓ | more KV streaming, less on-chip work | predicted |
| weight quantisation | ↑ | fewer bytes per FLOP, more compute per unit traffic | predicted |
| KV quantisation | ↑ | same, via `c_kv` | predicted |
| kernel fusion (SGLang, TRT-LLM) | ↑ | higher achieved BW and occupancy | predicted |
| tensor parallelism | ↓ | communication dilutes per-GPU work | predicted |

The TP prediction contradicts the intuition that more GPUs means more power
per GPU, and contradicts the paper's stated *reason* (they say arithmetic
intensity is unchanged; the actual mechanism is communication dilution).

---

## 5. What a publishable version needs

### Hardware diversity (blocking)

One card is a conjecture. Minimum set:

| card | TDP | cap_min | ρ | expected regime |
|---|---|---|---|---|
| A100 80GB PCIe | 300 W | 150 W | 0.50 | II (measured) |
| A100 SXM | 400 W | ~100 W | ~0.25 | cleanest control: same arch, different TDP |
| H100 / H200 | 700 W | ~200 W | ~0.29 | I (paper) |
| L4 / L40S | 72 / 350 W | ? | ? | III predicted: small card, η → 1 |

The A100 PCIe vs SXM pair is the strongest single comparison available:
identical architecture, identical workload, different `(ρ, η)`.

### From explanation to prediction

The criterion currently explains two points. To be a contribution it must
predict an unseen one. Two tiers:

- **Quantitative**: model `η` from workload parameters (`N`, `B`, `L`, `c_kv`,
  dtype) and hardware constants (peak BW, the ≈85 pJ/byte path cost, idle
  floor), then predict `P_decode` on a held-out card or workload. Compare
  against measurement. This mirrors what makes *Breaking the Ice* work: their
  contribution is the white-box step-wise predictor, not the 22-model sweep.
- **Qualitative**: given `(ρ, η)`, classify into regime I / II / III. Lower
  bar, and existing data already covers two regimes.

### Methodology gaps

- **Repetitions.** Currently 2–3, with repeat spread up to 3.3%. The paper uses
  10–20. Key configurations need 10.
- **Error bars.** Every figure currently reports medians only. The Pareto front
  especially: without them, a 0.3% ordering looks like a finding.
- **Difference method.** The `double-len` estimator (time `max_tokens=N` against
  `max_tokens=2N`) cancels prefill exactly, needs no scaling factor, makes no
  thermal-equality assumption, and reports the KV depth it measured. Validated
  at one point (2.2% agreement with the prefill-subtraction method, in the
  direction predicted by the deeper KV). Worth presenting as a methodological
  contribution; needs validation at large batch and long context, where the old
  method degrades (prefill is 19% of the window at BS=32, ~50% at L=32K).
- **CPU energy.** RAPL if readable in this VM, otherwise an explicit limitation.
  Steady-state decode should be GPU-dominated, but that needs data, not
  assertion.
- **System boundary.** NVML is board-level; RAPL package excludes board, fans
  and PSU loss. Their sum is a lower bound on node power, not node power. Only
  IPMI/Redfish gives the facility-relevant number.

### Framing risk

The core claim is "the published conclusion does not generalise". That reads as
a negative result unless the criterion has predictive power. Lead with the
criterion, use the disagreement as motivation, not as the contribution.

A forward-looking angle that is genuinely strengthened by the data: the industry
is moving toward **more** compute-rich accelerators (B200 at 1000 W+), which
pushes `η` down and `ρ` up. The illusion the paper describes gets *worse* over
time — their result is not wrong, it is the leading edge of a trend, and the
criterion says where the boundary lies.

---

## 6. Measuring (ρ, η)

`ρ`, read-only, no privileges:

```bash
python dvfs_probe.py probe --gpu 0
# "settable range: 150 W .. 300 W" and "default limit: 300 W"
```

`η`: run the decode workload at stock settings and take decode-phase power.
Not `power_median_w` of the full window, which still contains prefill:

```
P_decode = E_decode / t_decode
E_decode = E_window2 - E_window1 * prefill_scale     (prefill_scale = 1.0 under double-len)
```

Cap engagement, directly from the hardware rather than inferred:

```
pct_swpowercap = fraction of 50 ms samples with the SwPowerCap throttle bit set
```

Read it together with the power limit actually in force. The cap sweep runs at
the swept limit; the clock sweep runs with the limit restored to default. A 0%
in the clock sweep and a 99% in the cap sweep are not in conflict — they face
different walls. `dvfs_probe` prints the active limit as the `lim` column for
exactly this reason.

---

## 7. Open questions

1. Does the criterion predict a held-out card, or only rationalise measured
   ones?
2. Does long context move A100 PCIe from regime II back to regime I?
3. Does `η` differ between VM and bare metal on the same hardware? Without a
   bare-metal control, virtualisation stays a limitation rather than a finding.
4. Does a better-fused backend (SGLang, TensorRT-LLM) raise `η` enough to change
   regime? This is what makes a backend comparison a contribution rather than an
   engineering report.
5. Under TP, does per-GPU `η` fall as predicted? Testable on two GPUs in one
   node; no multi-node environment needed for the mechanism.
6. Is there a third coordinate? `ρ` and `η` say whether the cap *reaches* the
   operating point. They say nothing about closed-loop behaviour: overshoot,
   clock hunting, and inter-GPU frequency dispersion under collective
   synchronisation. That may be where clock locking retains a real advantage,
   and it is invisible at single-GPU scale.
