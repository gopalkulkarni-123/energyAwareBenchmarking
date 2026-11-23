#!/usr/bin/env python3
"""Table and plot of the double-len ridge runs: step time and network bytes.

usage: python3 ridge_table.py [--runs ridge_runs] [--dmon dmon_all.txt]

Reads every ridge_runs/tp<TP>_bs<B>_r<R>.log (output of bench_vllm.py
--method double-len) and the dmon file (nvidia-smi dmon -s t -d 1 -o T).

Per run:
  decode steps   = out_len x reps   (window 2 minus window 1, per sequence)
  step time      = (t_full - t_prefill) / decode steps
  bytes, GPU g   = mean dmon rate in window 2 x its length
                 - mean dmon rate in window 1 x its length
  bytes per step = bytes / decode steps          (MB, raw, no background removed)

Per setting: mean and sample standard deviation over the repeats.
The TP=1 rows are the background: on GPU 0 they hold the host traffic of
decode (inputs, launches); there is no GPU-to-GPU traffic at TP=1.

Writes ridge_runs.csv (one row per run), ridge_summary.csv (one row per
setting) and ridge_measured.png.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import time


def read_dmon(path, base_date):
    """Return [(epoch, gpu, rx, tx)]. Times are local HH:MM:SS; a jump back of
    more than 12 h means the file crossed midnight."""
    day0 = time.mktime(time.strptime(base_date, "%Y-%m-%d"))
    rows, day, prev = [], 0, None
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 4:
            continue
        try:
            h, m, s = map(int, p[0].split(":"))
            gpu, rx, tx = p[1], int(p[2]), int(p[3])
        except ValueError:
            continue                      # '-' or a partial line
        sec = 3600 * h + 60 * m + s
        if prev is not None and sec < prev - 43200:
            day += 1
        prev = sec
        rows.append((day0 + 86400 * day + sec, gpu, rx, tx))
    return rows


def window_mean(rows, gpu, lo, hi):
    v = [(rx, tx) for t, g, rx, tx in rows if g == gpu and lo <= t <= hi]
    if not v:
        return None, None, 0
    return (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v), len(v))


def mean_sd(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="ridge_runs")
    ap.add_argument("--dmon", default="dmon_all.txt")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    runs = []
    for f in sorted(glob.glob(os.path.join(a.runs, "tp*_bs*_r*.log"))):
        m = re.search(r"tp(\d+)_bs(\d+)_r(\d+)\.log$", f)
        tp, bs, rep = map(int, m.groups())
        marks, info = {}, None
        for line in open(f):
            if line.startswith("MARK "):
                _, n, t = line.split()
                marks[n] = float(t)
            elif line.startswith("WORKLOAD_JSON "):
                info = json.loads(line.split(" ", 1)[1])
        if info is None or len(marks) < 4:
            print(f"skip {f}: no result")
            continue
        runs.append(dict(file=f, tp=tp, bs=bs, rep=rep, marks=marks, info=info))
    if not runs:
        raise SystemExit("no runs found")

    first = min(r["marks"]["prefill_start"] for r in runs)
    rows = read_dmon(a.dmon, time.strftime("%Y-%m-%d", time.localtime(first)))

    per_run = []
    for r in runs:
        mk, d = r["marks"], r["info"]
        w1 = (mk["prefill_start"], mk["prefill_end"])
        w2 = (mk["full_start"], mk["full_end"])
        steps = d["out_len"] * d["decode_reps"]
        row = dict(tp=r["tp"], bs=r["bs"], rep=r["rep"], steps=steps,
                   tok_per_s=d["decode_tok_per_s"],
                   step_ms=1000 * d["t_decode_s"] / steps)
        for g in ("0", "1"):
            rx1, tx1, n1 = window_mean(rows, g, *w1)
            rx2, tx2, n2 = window_mean(rows, g, *w2)
            row[f"n1_gpu{g}"], row[f"n2_gpu{g}"] = n1, n2
            if n1 and n2:
                T1, T2 = w1[1] - w1[0], w2[1] - w2[0]
                row[f"tx_mb_gpu{g}"] = (tx2 * T2 - tx1 * T1) / steps
                row[f"rx_mb_gpu{g}"] = (rx2 * T2 - rx1 * T1) / steps
            else:
                row[f"tx_mb_gpu{g}"] = row[f"rx_mb_gpu{g}"] = None
        per_run.append(row)

    os.makedirs(a.out, exist_ok=True)
    cols = list(per_run[0].keys())
    with open(os.path.join(a.out, "ridge_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in sorted(per_run, key=lambda x: (x["tp"], x["bs"], x["rep"])):
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in row.items()})

    summary = []
    for key in sorted({(x["tp"], x["bs"]) for x in per_run}):
        sel = [x for x in per_run if (x["tp"], x["bs"]) == key]
        s = dict(tp=key[0], bs=key[1], runs=len(sel))
        for q in ("tok_per_s", "step_ms", "tx_mb_gpu0", "tx_mb_gpu1",
                  "rx_mb_gpu0", "rx_mb_gpu1"):
            s[q], s[q + "_sd"] = mean_sd([x[q] for x in sel])
        summary.append(s)
    with open(os.path.join(a.out, "ridge_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for s in summary:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in s.items()})

    def fmt(m, sd, nd=3):
        return "        -" if m is None else f"{m:.{nd}f} ± {sd:.{nd}f}"

    print(f"{'setting':<11}{'runs':>5}  {'tok/s':>15}  {'ms per step':>15}  "
          f"{'TX MB/step GPU0':>17}  {'TX MB/step GPU1':>17}")
    for s in summary:
        print(f"tp{s['tp']} bs{s['bs']:<4}{s['runs']:>6}  "
              f"{fmt(s['tok_per_s'], s['tok_per_s_sd'], 1):>15}  "
              f"{fmt(s['step_ms'], s['step_ms_sd']):>15}  "
              f"{fmt(s['tx_mb_gpu0'], s['tx_mb_gpu0_sd']):>17}  "
              f"{fmt(s['tx_mb_gpu1'], s['tx_mb_gpu1_sd']):>17}")
    print("(mean ± sample standard deviation over the repeats; MB as dmon reports it;"
          " no background removed, the tp1 row on GPU 0 is the background)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker
    except ImportError:
        print("matplotlib not installed: table written, no plot")
        return

    ink, muted, grid = "#2b2b29", "#6b6b66", "#e4e3df"
    c0, c1 = "#2a78d6", "#eb6834"          # GPU 0, GPU 1 (validated pair)
    tp2 = [s for s in summary if s["tp"] == 2]
    tp1 = [s for s in summary if s["tp"] == 1 and s["bs"] == 1]
    xs = [s["bs"] for s in tp2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel("batch size (sequences decoded together)", color=ink)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(muted)
        ax.tick_params(colors=muted)

    ax1.errorbar(xs, [s["step_ms"] for s in tp2], yerr=[s["step_ms_sd"] for s in tp2],
                 color=c0, marker="o", markersize=7, linewidth=2, capsize=3,
                 label="TP=2")
    if tp1:
        ax1.axhline(tp1[0]["step_ms"], color=muted, linestyle="--", linewidth=1.5)
        ax1.annotate(f"TP=1, batch 1: {tp1[0]['step_ms']:.2f} ms", (xs[0], tp1[0]["step_ms"]),
                     xytext=(4, 5), textcoords="offset points", color=muted, fontsize=9)
    ax1.set_ylabel("time per decode step (ms)", color=ink)
    ax1.set_ylim(bottom=0)
    ax1.set_title("Decode step time, TP=2", color=ink, loc="left")

    for g, c in (("0", c0), ("1", c1)):
        ys = [s[f"tx_mb_gpu{g}"] for s in tp2]
        ax2.errorbar(xs, ys, yerr=[s[f"tx_mb_gpu{g}_sd"] for s in tp2], color=c,
                     marker="o", markersize=7, linewidth=2, capsize=3,
                     label=f"GPU {g}")
        if ys[-1] is not None:
            ax2.annotate(f"GPU {g}", (xs[-1], ys[-1]), xytext=(6, -4 if g == "0" else 6),
                         textcoords="offset points", color=ink, fontsize=9)
    if tp1 and tp1[0]["tx_mb_gpu0"] is not None:
        ax2.axhline(tp1[0]["tx_mb_gpu0"], color=muted, linestyle="--", linewidth=1.5)
        ax2.annotate("TP=1 background (GPU 0)", (xs[0], tp1[0]["tx_mb_gpu0"]),
                     xytext=(4, 5), textcoords="offset points", color=muted, fontsize=9)
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax2.set_ylabel("MB sent per decode step (PCIe TX)", color=ink)
    ax2.set_title("Network bytes per decode step, TP=2", color=ink, loc="left")
    ax2.legend(frameon=False, loc="upper left")

    fig.text(0.01, 0.01, "Qwen3-4B, A100 80GB PCIe x2, double-len, in 1024 / out 512. "
             "Points: mean of repeats; bars: ±1 standard deviation.", color=muted, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(a.out, "ridge_measured.png")
    fig.savefig(out, dpi=150)
    print(f"wrote ridge_runs.csv, ridge_summary.csv, {out}")


if __name__ == "__main__":
    main()
