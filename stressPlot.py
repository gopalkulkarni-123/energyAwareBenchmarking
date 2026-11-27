#!/usr/bin/env python3
"""Stress test plot: decode throughput and time per token against batch size.

usage: python3 stress_plot.py [results/stress] [--knee 128] [--out results/stress]

Reads every bs*.log in the folder (WORKLOAD_JSON line and vLLM's
"Maximum concurrency ... : N x" line). Two panels, same x axis:
  top    : total decode throughput (tokens/s), knee marked
  bottom : time per decode step = time each user waits per token (ms)
Batch sizes above vLLM's KV-cache capacity are shaded: they cannot all run at once.
Writes stress_plot.png and stress_summary.csv.
"""
import argparse
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(folder):
    rows = []
    for f in glob.glob(os.path.join(folder, "bs*.log")):
        txt = open(f, errors="replace").read()
        m = re.search(r"WORKLOAD_JSON (\{.*\})", txt)
        if not m:
            print(f"{f}: no result, skipped")
            continue
        w = json.loads(m.group(1))
        cap = re.findall(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", txt)
        n_steps = w["n_decode_tokens"] / w["bs"]
        rows.append(dict(bs=w["bs"], tok_s=w["decode_tok_per_s"],
                         ms_step=1000 * w["t_decode_s"] / n_steps,
                         cap=float(cap[-1]) if cap else None,
                         preempt=len(re.findall(r"preempt", txt, re.I))))
    return sorted(rows, key=lambda r: r["bs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="results/stress")
    ap.add_argument("--knee", type=int, default=None, help="batch to mark (default: max throughput)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or a.folder
    rows = read(a.folder)
    if not rows:
        raise SystemExit("no results found")

    knee = a.knee or max(rows, key=lambda r: r["tok_s"])["bs"]
    caps = [r["cap"] for r in rows if r["cap"]]
    cap = min(caps) if caps else None

    print(f"{'batch':>6}{'tok/s':>9}{'ms/step':>9}{'cap':>8}{'preempt':>9}")
    for r in rows:
        print(f"{r['bs']:6d}{r['tok_s']:9.0f}{r['ms_step']:9.1f}"
              f"{(r['cap'] or 0):8.0f}{r['preempt']:9d}")
    print(f"knee at batch {knee}; smallest vLLM capacity {cap:.0f} requests" if cap else
          f"knee at batch {knee}")

    with open(os.path.join(out, "stress_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ink, muted, grid, blue, orange = "#2b2b29", "#6b6b66", "#e4e3df", "#2a78d6", "#c4501b"
    xs = [r["bs"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    ax1.plot(xs, [r["tok_s"] for r in rows], "o-", color=blue, linewidth=2, markersize=7)
    ax2.plot(xs, [r["ms_step"] for r in rows], "s-", color=orange, linewidth=2, markersize=6)
    for ax in (ax1, ax2):
        ax.axvline(knee, color=ink, linestyle="--", linewidth=1)
        if cap:
            ax.axvspan(cap, max(xs) + 16, color="#eeeeec", zorder=0)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(muted)
        ax.tick_params(colors=muted)
    kr = next(r for r in rows if r["bs"] == knee)
    ax1.annotate(f"knee: batch {knee}\n{kr['tok_s']:,.0f} tokens/s", (knee, kr["tok_s"]),
                 xytext=(10, -40), textcoords="offset points", color=ink, fontsize=9)
    if cap:
        ax1.text(cap + 3, ax1.get_ylim()[0] + 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                 f"above KV-cache capacity\n(about {cap:.0f} requests)", color=muted, fontsize=8)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.set_ylabel("Decode throughput (tokens/s)", color=ink)
    ax2.set_ylabel("Time per token per user (ms)", color=ink)
    ax2.set_xlabel("Batch size (concurrent requests)", color=ink)
    ax2.set_xticks(xs)
    ax2.set_xlim(min(xs) - 16, max(xs) + 16)
    ax1.set_title("Qwen3-4B decode on one A100 80GB PCIe: stress test (default clock)",
                  color=ink, loc="left", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out, "stress_plot.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path} and stress_summary.csv")


if __name__ == "__main__":
    main()
