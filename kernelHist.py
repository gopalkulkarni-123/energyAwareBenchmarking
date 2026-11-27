#!/usr/bin/env python3
"""Bar chart of every kernel in one decode step: share of step time, per kernel.

usage: python3 kernel_hist.py kernel_list_ncu_bs1.csv [kernel_list_ncu_bs64.csv ...]
       [--metric time|bytes|flops] [--top 40]

Input: the kernel_list_<file>.csv written by kernelList.py (one row per unique
kernel name). Every kernel gets its own bar, sorted from largest to smallest;
the colour shows its group (matmul, attention, other), and each bar is
labelled with its share and the running total. Kernels beyond --top are summed into one last bar.
Writes kernel_hist_<metric>_<file>.png next to each input and prints the table.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"time": ("t_ms", "share of step time (%)"),
       "bytes": ("GB", "share of HBM bytes (%)"),
       "flops": ("GFLOP", "share of executed FLOPs (%)")}
COLORS = {"matmul": "#2a78d6", "attention": "#c4501b", "other": "#8a94a6"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--metric", choices=COL, default="time")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    key, ylabel = COL[a.metric]
    ink, muted, grid = "#2b2b29", "#6b6b66", "#e4e3df"

    for path in a.csv:
        rows = list(csv.DictReader(open(path)))
        for r in rows:
            r["v"] = float(r[key])
            r["launches"] = int(r["launches"])
        tot = sum(r["v"] for r in rows) or 1.0
        rows.sort(key=lambda r: -r["v"])
        show = rows[: a.top]
        rest = rows[a.top:]

        print(f"\n{os.path.basename(path)}: {len(rows)} unique kernels, "
              f"{sum(r['launches'] for r in rows)} launches, metric = {a.metric}")
        print(f"{'#':>3} {'group':<10}{'launches':>9}{'share':>8}{'cumul.':>8}  kernel")
        cum = 0.0
        for i, r in enumerate(rows, 1):
            cum += r["v"] / tot
            print(f"{i:3d} {r['group']:<10}{r['launches']:9d}{100*r['v']/tot:7.2f}%"
                  f"{100*cum:7.1f}%  {r['short']}")
        for g in ("matmul", "attention", "other"):
            s = sum(r["v"] for r in rows if r["group"] == g)
            n = sum(1 for r in rows if r["group"] == g)
            print(f"    {g:<10} {n:3d} kernels  {100*s/tot:6.2f}% of {a.metric}")

        labels = [r["short"][:48] for r in show]
        vals = [100 * r["v"] / tot for r in show]
        cols = [COLORS.get(r["group"], "#8a94a6") for r in show]
        if rest:
            labels.append(f"{len(rest)} smaller kernels")
            vals.append(100 * sum(r["v"] for r in rest) / tot)
            cols.append("#cfd3da")
        fig, ax = plt.subplots(figsize=(11, 0.28 * len(vals) + 1.6))
        y = range(len(vals))
        ax.barh(y, vals, color=cols, height=0.75)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=7.5, color=ink)
        ax.invert_yaxis()
        cum = 0.0
        for i, v in enumerate(vals):
            cum += v
            ax.text(v, i, f"  {v:.1f}%  (total so far {cum:.0f}%)", va="center",
                    fontsize=7, color=muted)
        ax.set_xlim(0, max(vals) * 1.45)
        ax.set_xlabel(ylabel, color=ink)
        ax.grid(True, axis="x", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("right",):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="x", colors=muted)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
        ax.legend(handles, list(COLORS), frameon=False, loc="lower right", fontsize=8)
        name = os.path.splitext(os.path.basename(path))[0].replace("kernel_list_", "")
        ax.set_title(f"Every kernel in one decode step ({name}), by {a.metric}",
                     color=ink, loc="left", fontsize=10)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(path) or ".", f"kernel_hist_{a.metric}_{name}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
