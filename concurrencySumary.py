#!/usr/bin/env python3
"""Summarize the maximum-concurrency test (active requests per engine step).

usage: python3 concurrency_summary.py [results/concurrency] [--out-len 1024]

Reads, for every batch size N in the folder:
  bsN.log         vLLM's KV-cache line ("GPU KV cache size: X tokens,
                  Maximum concurrency for L tokens per request: Cx")
  active_bsN.csv  columns step, active (requests that got a token in that step)

Writes:
  concurrency_summary.csv   one row per batch size (columns explained below)
  concurrency_active.png    active requests per step, one panel per batch size,
                            with vLLM's predicted capacity as a dashed line

Columns:
  bs                   requests sent
  kv_pool_tokens       KV-cache pool, from vLLM's log
  max_len              tokens per request used for vLLM's capacity number
  vllm_capacity        pool / max_len, vLLM's predicted maximum concurrency
  steps                engine steps the run took
  extra_steps          steps - out_len; about 0 to 50 is prefill and finishing,
                       much more means requests were paused and recomputed
  fits_by_memory       yes if bs x (in_len + out_len) <= pool, i.e. all requests
                       fit at their full length
  predicted_drop_step  decode step where the pool runs out: pool / bs - in_len
                       (empty if it never runs out)
  measured_drop_step   first step, after all requests are running, where fewer
                       than bs got a token. For a batch that fits this is the end
                       of the run (requests finishing), about out_len.
  full_pct             share of steps from full start to out_len that served all
"""
import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="results/concurrency")
    ap.add_argument("--in-len", type=int, default=1024)
    ap.add_argument("--out-len", type=int, default=1024)
    a = ap.parse_args()

    rows, series = [], {}
    for path in glob.glob(os.path.join(a.folder, "active_bs*.csv")):
        bs = int(re.search(r"active_bs(\d+)\.csv", path).group(1))
        act = [int(r["active"]) for r in csv.DictReader(open(path))]
        log = os.path.join(a.folder, f"bs{bs}.log")
        pool = max_len = cap = None
        if os.path.exists(log):
            m = re.findall(r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for "
                           r"([\d,]+) tokens per request: ([\d.]+)x", open(log, errors="replace").read())
            if m:
                pool, max_len, cap = (int(m[-1][0].replace(",", "")),
                                      int(m[-1][1].replace(",", "")), float(m[-1][2]))
        start = next((i for i, x in enumerate(act) if x == bs), 0)
        drop = next((i for i in range(start, len(act)) if act[i] < bs), len(act))
        pred = None
        if pool and pool // bs < a.in_len + a.out_len:
            pred = pool // bs - a.in_len
        win = act[start:max(start + 1, min(a.out_len, len(act)))]
        rows.append(dict(bs=bs, kv_pool_tokens=pool, max_len=max_len, vllm_capacity=cap,
                         steps=len(act), extra_steps=len(act) - a.out_len,
                         fits_by_memory=("yes" if pred is None else "no") if pool else "?",
                         predicted_drop_step=pred if pred is not None else "",
                         measured_drop_step=drop,
                         full_pct=round(100 * sum(x == bs for x in win) / len(win), 1)))
        series[bs] = act
    if not rows:
        raise SystemExit(f"no active_bs*.csv files in {a.folder}")
    rows.sort(key=lambda r: r["bs"])

    cols = list(rows[0].keys())
    with open(os.path.join(a.folder, "concurrency_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{'bs':>5}{'pool':>10}{'vLLM cap':>10}{'steps':>7}{'extra':>7}{'full %':>8}"
          f"  {'fits':<5}{'predicted drop':>15}{'measured drop':>15}")
    for r in rows:
        print(f"{r['bs']:5d}{(r['kv_pool_tokens'] or 0):10,d}{(r['vllm_capacity'] or 0):10.1f}"
              f"{r['steps']:7d}{r['extra_steps']:7d}{r['full_pct']:8.1f}  {r['fits_by_memory']:<5}"
              f"{str(r['predicted_drop_step'] or 'never'):>15}{r['measured_drop_step']:>15}")
    ok = [r["bs"] for r in rows if r["fits_by_memory"] == "yes"]
    no = [r["bs"] for r in rows if r["fits_by_memory"] == "no"]
    if ok and no:
        print(f"\nmemory limit: between {max(ok)} and {min(no)} concurrent requests "
              f"at {a.in_len + a.out_len} tokens each")

    ink, muted, grid, blue, orange = "#2b2b29", "#6b6b66", "#e4e3df", "#2a78d6", "#c4501b"
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.9 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        act = series[r["bs"]]
        ax.plot(range(len(act)), act, color=blue if r["fits_by_memory"] == "yes" else orange,
                linewidth=1.2)
        if r["predicted_drop_step"] != "":
            ax.axvline(r["predicted_drop_step"], color=orange, linestyle="--", linewidth=1)
        ax.axhline(r["bs"], color=muted, linewidth=0.8)
        if r["vllm_capacity"]:
            ax.axhline(r["vllm_capacity"], color=ink, linestyle="--", linewidth=1)
        ax.axvline(a.out_len, color=muted, linestyle=":", linewidth=1)
        ax.set_ylim(0, max(r["bs"], r["vllm_capacity"] or 0) * 1.12)
        pd = r["predicted_drop_step"]
        ax.text(0.01, 0.08, f"batch {r['bs']}: memory runs out at step "
                f"{pd if pd != '' else 'never'} (predicted), count first drops at step "
                f"{r['measured_drop_step']} (measured), {r['extra_steps']} extra steps",
                transform=ax.transAxes, fontsize=8, color=ink)
        ax.set_ylabel("active", color=ink, fontsize=9)
        ax.grid(True, color=grid, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)
    axes[-1].set_xlabel("engine step", color=ink)
    axes[0].set_title(f"Requests served per step. Solid grey: requests sent. Dashed: vLLM's "
                      f"predicted capacity. Dotted: {a.out_len} steps (ideal run length). "
                      f"Orange dashed: predicted step where memory runs out.",
                      color=ink, loc="left", fontsize=9)
    fig.tight_layout()
    out = os.path.join(a.folder, "concurrency_active.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote concurrency_summary.csv and concurrency_active.png in {a.folder}")


if __name__ == "__main__":
    main()