#!/usr/bin/env python3
"""
extract.py -- turn dvfs_probe sweep CSVs into a tidy per-configuration table
of decode-phase metrics.

Decode is not measured directly: bench_vllm.py runs a prefill-only window and
a prefill+decode window, and decode is the difference. Every metric below
applies that same subtraction, then normalises by the number of decode tokens
so that configurations which auto-calibrated to different repetition counts
stay comparable.

    E_decode = E_full - E_prefill * prefill_scale
    t_decode = t_full - t_prefill * prefill_scale
    J/token    = E_decode / n_decode_tokens
    ms/token   = 1000 * t_decode / n_decode_tokens

Emits decode_metrics.csv with one row per (model, batch, mode, setting).
"""
import csv, glob, os, re, statistics as st

SRC = os.environ.get("SRC", "results/dvfs/csv")

# The filename is not a reliable key: one of the uploads is the 8B model but
# still matches a bs1 pattern. Read model/bs out of the CSV contents instead.
def sweeps():
    for path in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
        rows = list(csv.DictReader(open(path)))
        if not rows or "phase" not in rows[0]:
            continue
        yield path, rows

def num(r, k, default=None):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return default

out = []
for path, rows in sweeps():
    model = os.path.basename(rows[0].get("model", "?").rstrip("/"))
    bs = int(num(rows[0], "bs", 0))
    for mode in ("cap", "clock"):
        settings = sorted({r["setting"] for r in rows if r["mode"] == mode},
                          key=float)
        for s in settings:
            F = [r for r in rows if r["mode"] == mode and r["setting"] == s
                 and r["phase"] == "full"]
            P = [r for r in rows if r["mode"] == mode and r["setting"] == s
                 and r["phase"] == "prefill"]
            if not F or len(F) != len(P):
                continue
            jt, mst, pw, sm, cap, bw = [], [], [], [], [], []
            for f, p in zip(F, P):
                scale = num(f, "prefill_scale", 0.0)
                ntok = num(f, "n_decode_tokens")
                if not ntok:
                    continue
                e_dec = num(f, "energy_trapz_j") - num(p, "energy_trapz_j") * scale
                t_dec = num(f, "t_decode_s")          # bench_vllm already subtracts
                jt.append(e_dec / ntok)
                mst.append(1000.0 * t_dec / ntok)
                # Decode-phase power, reconstructed from the decode-only
                # energy and time rather than read off the full window, which
                # still contains the prefill slice.
                pw.append(e_dec / t_dec if t_dec > 0 else float("nan"))
                sm.append(num(f, "sm_median_mhz"))
                cap.append(num(f, "pct_swpowercap"))
            if not jt:
                continue
            out.append(dict(
                model=model, bs=bs, mode=mode, setting=int(float(s)),
                j_per_tok=round(st.median(jt), 4),
                ms_per_tok=round(st.median(mst), 4),
                tok_per_s=round(1000.0 / st.median(mst), 2),
                decode_w=round(st.median(pw), 1),
                sm_mhz=int(st.median(sm)),
                cap_pct=round(st.median(cap), 1),
                reps=len(jt),
            ))

keys = ["model", "bs", "mode", "setting", "j_per_tok", "ms_per_tok",
        "tok_per_s", "decode_w", "sm_mhz", "cap_pct", "reps"]
with open("decode_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(out)
print(f"wrote decode_metrics.csv ({len(out)} rows)")
for r in out:
    print(f"{r['model']:10} bs={r['bs']:<3} {r['mode']:5} {r['setting']:>5} "
          f"| {r['j_per_tok']:>8.3f} J/tok | {r['ms_per_tok']:>7.3f} ms/tok "
          f"| {r['decode_w']:>6.1f} W | SM {r['sm_mhz']:>4} | cap {r['cap_pct']:>5.1f}%")
