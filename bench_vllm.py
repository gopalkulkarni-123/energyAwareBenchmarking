#!/usr/bin/env python3
"""
bench_vllm.py -- phase-separated vLLM prefill/decode benchmark, designed to be
called by dvfs_probe.py via --workload

It does three things:
  1. Builds the batch from fixed-length token ids so prompt length is exact.
  2. Issues two separate generate calls: max_tokens=1 (prefill only) and
     max_tokens=N (prefill + decode). Decode energy comes from the difference:
         E_decode = E_full - E_prefill * (decode_reps / prefill_reps)
  3. Emits window boundaries as "MARK <name> <epoch>" on stdout so dvfs_probe
     can exclude model loading, CUDA graph capture and warmup from the
     integration interval.

It also runs standalone; you just get no power data that way.

Examples:
  python bench_vllm.py --model Qwen/Qwen3-4B --bs 1  --in-len 1024 --out-len 1024
  python bench_vllm.py --model Qwen/Qwen3-4B --bs 32 --in-len 4096 --out-len 256

Together with dvfs_probe:
  sudo python dvfs_probe.py sweep \
       --workload "python bench_vllm.py --model Qwen/Qwen3-4B --bs 32 --in-len 4096 --out-len 256" \
       --reps 3 --out qwen_bs32.csv
"""

import argparse
import json
import os
import random
import sys
import time


def mark(name):
    print(f"MARK {name} {time.time():.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--bs", type=int, default=1, help="batch size (concurrent sequences)")
    ap.add_argument("--in-len", type=int, default=1024, help="tokens per prompt")
    ap.add_argument("--out-len", type=int, default=128, help="tokens generated per sequence")
    ap.add_argument("--prefill-reps", type=int, default=0,
                    help="repetitions inside the prefill window. "
                         "0 = auto-calibrate against --target-window-s")
    ap.add_argument("--decode-reps", type=int, default=0,
                    help="repetitions inside the full window. "
                         "0 = auto-calibrate against --target-window-s")
    ap.add_argument("--target-window-s", type=float, default=8.0,
                    help="target window duration for auto-calibration. "
                         "At 50 ms sampling, 8 s gives roughly 160 points")
    ap.add_argument("--max-reps", type=int, default=1000)
    ap.add_argument("--method", default="prefill-sub",
                    choices=("prefill-sub", "double-len"),
                    help="How decode is isolated. 'prefill-sub' (default) times a "
                         "prefill-only window and subtracts it, scaled by the "
                         "repetition ratio. 'double-len' instead times "
                         "max_tokens=N and max_tokens=2N: both contain exactly "
                         "one identically shaped prefill, so prefill cancels "
                         "exactly with no scaling and no assumption that its "
                         "thermal state matches. Use it whenever prefill is a "
                         "large fraction of the full window, which is what "
                         "happens at long context (prefill attention is O(L^2), "
                         "decode is O(L)) and at large batch.")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--eager", action="store_true",
                    help="enforce_eager. Off by default for DVFS measurement so the "
                         "run stays close to production; turn it on for NCU roofline work")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--settle-s", type=float, default=1.0,
                    help="quiet time between windows so the power curve settles and "
                         "windows do not bleed into each other")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    try:
        import vllm
        ver = vllm.__version__
    except Exception:
        ver = "unknown"

    # double-len generates up to 2*out_len tokens, so the context budget has to
    # cover the longer of the two windows or vLLM truncates the long one.
    _need = args.in_len + (2 if args.method == "double-len" else 1) * args.out_len + 64
    max_len = args.max_model_len or _need
    if max_len < _need:
        print(f"[warn] --max-model-len {max_len} is below the {_need} tokens this "
              f"configuration needs; the longer window will be truncated",
              file=sys.stderr)
    batched = args.bs * args.in_len

    # Chunked prefill slices the prefill and interleaves it with decode steps,
    # which would make the prefill window impure. As long as
    # max_num_batched_tokens >= bs*in_len, no chunking actually happens.
    # Above 128K we give up (it would OOM): prefill numbers then include
    # chunking effects, but decode numbers stay valid.
    if batched <= 131072:
        max_batched = batched
        chunk_warn = None
    else:
        max_batched = None
        chunk_warn = (f"bs*in_len = {batched} > 131072, not forcing chunked prefill off. "
                      "Prefill-phase numbers will include chunking effects; "
                      "the decode phase is unaffected.")

    kw = dict(
        model=args.model,
        dtype=args.dtype,
        max_model_len=max_len,
        max_num_seqs=max(args.bs, 1),
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=False,   # critical: otherwise the second generate is a cache hit
        enforce_eager=args.eager,
        disable_log_stats=True,
        seed=args.seed,
    )
    if max_batched is not None:
        kw["max_num_batched_tokens"] = max_batched

    t_load0 = time.time()
    llm = LLM(**kw)
    t_load = time.time() - t_load0

    tok = llm.get_tokenizer()
    vocab = getattr(tok, "vocab_size", None) or len(tok)
    rng = random.Random(args.seed)

    # Distinct random tokens per sequence, ruling out any form of prefix reuse
    prompts = [
        {"prompt_token_ids": [rng.randrange(100, vocab - 100) for _ in range(args.in_len)]}
        for _ in range(args.bs)
    ]

    sp_prefill = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0.0)
    sp_full = SamplingParams(max_tokens=args.out_len, ignore_eos=True, temperature=0.0)
    # double-len pairs max_tokens=N against max_tokens=2N.
    sp_long = SamplingParams(max_tokens=2 * args.out_len, ignore_eos=True,
                             temperature=0.0)
    dbl = args.method == "double-len"

    # ---- warmup: must use exactly the same shapes as the measurement, otherwise
    #      Triton kernels JIT-compile near the measurement window (see the
    #      _compute_slot_mapping_kernel warning in earlier logs) ----
    for _ in range(args.warmup):
        llm.generate(prompts, sp_prefill if not dbl else sp_full, use_tqdm=False)
        llm.generate(prompts, sp_long if dbl else sp_full, use_tqdm=False)

    # ---- calibration: derive repetition counts from a single timed iteration ----
    def _time_once(sp):
        t = time.time()
        llm.generate(prompts, sp, use_tqdm=False)
        return time.time() - t

    def _calib(sp, explicit):
        """Auto-calibrate repetitions to reach --target-window-s.

        Note the discretisation: n = int(target/dt)+1 flips between 1 and 2
        right at dt == target, so two settings whose per-call time straddles
        that threshold get windows differing by 2x for no physical reason.
        window_s is therefore NOT comparable across settings -- always compare
        per-call or per-token figures (decode_tok_per_s, n_decode_tokens).
        For a sweep where comparable windows matter, pin --prefill-reps and
        --decode-reps explicitly instead of relying on this.
        """
        if explicit > 0:
            return explicit, None
        dt = _time_once(sp)
        n = max(1, min(args.max_reps, int(args.target_window_s / max(dt, 1e-6)) + 1))
        return n, round(dt, 5)

    if dbl:
        # Both windows MUST use the same repetition count, otherwise the
        # prefill contributions no longer cancel and we are back to scaling.
        reps, t_once = _calib(sp_long, args.decode_reps)
        short_reps = long_reps = reps
        print(f"# calib: double-len, {t_once}s per long call x{reps} "
              f"(same reps for both windows)", file=sys.stderr)
    else:
        short_reps, t_once = _calib(sp_prefill, args.prefill_reps)
        long_reps, t_full_once = _calib(sp_full, args.decode_reps)
        print(f"# calib: prefill {t_once}s x{short_reps} | "
              f"full {t_full_once}s x{long_reps}", file=sys.stderr)

    time.sleep(args.settle_s)

    # ---- window 1 ----
    # prefill-sub: prefill only (plus one decode step)
    # double-len:  one prefill plus N decode tokens
    sp1 = sp_full if dbl else sp_prefill
    mark("prefill_start")
    t0 = time.time()
    for _ in range(short_reps):
        llm.generate(prompts, sp1, use_tqdm=False)
    t_short = time.time() - t0
    mark("prefill_end")

    time.sleep(args.settle_s)

    # ---- window 2 ----
    # prefill-sub: one prefill plus the full decode
    # double-len:  one prefill plus 2N decode tokens
    sp2 = sp_long if dbl else sp_full
    mark("full_start")
    t0 = time.time()
    outs = None
    for _ in range(long_reps):
        outs = llm.generate(prompts, sp2, use_tqdm=False)
    t_long = time.time() - t0
    mark("full_end")

    for nm, w in (("window 1", t_short), ("window 2", t_long)):
        if w < 3.0:
            print(f"[warn] {nm} is only {w:.2f}s, too few samples; "
                  f"consider raising --target-window-s or --out-len", file=sys.stderr)

    # ---- reconcile the number of tokens actually generated ----
    gen = sum(len(o.token_ids) for out in outs for o in out.outputs)

    if dbl:
        # The difference is decode tokens N+1..2N, at KV depth in_len+N to
        # in_len+2N. prefill_scale is exactly 1: one identical prefill on each
        # side cancels with no adjustment.
        n_decode = args.bs * args.out_len * reps
        scale = 1.0
        kv_lo = args.in_len + args.out_len
        kv_hi = args.in_len + 2 * args.out_len
    else:
        n_decode = args.bs * (args.out_len - 1) * long_reps
        scale = long_reps / short_reps
        kv_lo = args.in_len
        kv_hi = args.in_len + args.out_len

    t_decode = t_long - t_short * scale

    info = dict(
        vllm_version=ver,
        model=args.model,
        method=args.method,
        bs=args.bs,
        in_len=args.in_len,
        out_len=args.out_len,
        eager=args.eager,
        load_s=round(t_load, 2),
        prefill_reps=short_reps,
        decode_reps=long_reps,
        t_prefill_s=round(t_short, 4),
        t_full_s=round(t_long, 4),
        # E_decode = E_window2 - E_window1 * prefill_scale
        prefill_scale=round(scale, 6),
        t_decode_s=round(t_decode, 4),
        n_prefill_tokens=args.bs * args.in_len * short_reps,
        n_gen_per_rep=gen,
        n_decode_tokens=n_decode,
        kv_depth_lo=kv_lo,
        kv_depth_hi=kv_hi,
    )
    info["decode_tok_per_s"] = round(n_decode / max(t_decode, 1e-9), 2)
    if chunk_warn:
        info["warning"] = chunk_warn
        print(f"[warn] {chunk_warn}", file=sys.stderr)

    print("WORKLOAD_JSON " + json.dumps(info), flush=True)
    print(f"WORKLOAD_METRIC {n_decode}", flush=True)   # throughput proxy = decode tokens

    # Human-readable summary (dvfs_probe ignores these lines)
    print(f"# load {t_load:.1f}s | {args.method} | w1 {t_short:.2f}s x{short_reps} | "
          f"w2 {t_long:.2f}s x{long_reps} | decode {t_decode:.2f}s "
          f"({info['decode_tok_per_s']} tok/s, KV depth {kv_lo}-{kv_hi})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
