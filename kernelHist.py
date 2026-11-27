#!/usr/bin/env python3
"""Bar chart of every kernel in one decode step: share of step time, per kernel.
 
usage: python3 kernel_hist.py kernel_list_ncu_bs1.csv [kernel_list_ncu_bs64.csv ...]
       [--metric time|bytes|flops|perf|bw] [--top 40]
 
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
       "flops": ("GFLOP", "share of executed FLOPs (%)"),
       "perf": (None, "performance, executed TFLOP/s"),
       "bw": (None, "achieved HBM bandwidth (TB/s)")}
ROOF = {"perf": (232.8, "measured compute roof 232.8 TFLOP/s"),
        "bw": (1.695, "measured memory roof 1.695 TB/s")}
COLORS = {"matmul": "#2a78d6", "attention": "#c4501b", "other": "#8a94a6"}
 
 
def plot_abs(rows, a, path, T):
    """perf or bw: absolute value per kernel, largest kernels by time first."""
    ink, muted, grid = "#2b2b29", "#6b6b66", "#e4e3df"
    roof, rlabel = ROOF[a.metric]
    unit = "TFLOP/s" if a.metric == "perf" else "TB/s"
    print(f"\n{os.path.basename(path)}: kernels with >= {a.min_share}% of step time, "
          f"largest first")
    print(f"{'group':<10}{'% time':>8}{unit:>10}{'% of roof':>10}  kernel")
    for r in rows:
        print(f"{r['group']:<10}{100*float(r['t_ms'])/T:7.1f}%{r['v']:10.3f}"
              f"{100*r['v']/roof:9.1f}%  {r['short']}")
    labels = [f"{r['short'][:44]}  ({100*float(r['t_ms'])/T:.0f}% of time)" for r in rows]
    vals = [r["v"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(vals) + 1.6))
    y = list(range(len(vals)))
    ax.barh(y, vals, color=[COLORS.get(r["group"], "#8a94a6") for r in rows], height=0.75)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.2f} {unit} ({100*v/roof:.0f}% of roof)", va="center",
                fontsize=7, color=muted)
    ax.axvline(roof, color=ink, linestyle="--", linewidth=1)
    ax.text(roof, -0.9, rlabel + " ", ha="right", fontsize=7.5, color=ink)
    ax.set_xlim(0, max(roof, max(vals)) * 1.25)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5, color=ink)
    ax.invert_yaxis()
    ax.set_xlabel(COL[a.metric][1], color=ink)
    ax.grid(True, axis="x", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    ax.legend(handles, list(COLORS), frameon=False, loc="lower right", fontsize=8)
    name = os.path.splitext(os.path.basename(path))[0].replace("kernel_list_", "")
    ax.set_title(f"Kernels in one decode step ({name}): {COL[a.metric][1]}",
                 color=ink, loc="left", fontsize=10, pad=18)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(path) or ".", f"kernel_hist_{a.metric}_{name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--metric", choices=COL, default="time")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-share", type=float, default=0.5,
                    help="perf/bw only: show kernels with at least this %% of step time")
    a = ap.parse_args()
    key, ylabel = COL[a.metric]
    ink, muted, grid = "#2b2b29", "#6b6b66", "#e4e3df"
 
    for path in a.csv:
        rows = list(csv.DictReader(open(path)))
        for r in rows:
            r["launches"] = int(r["launches"])
            t = float(r["t_ms"]) / 1e3
            if a.metric == "perf":
                r["v"] = float(r["GFLOP"]) / 1e3 / t if t else 0.0
            elif a.metric == "bw":
                r["v"] = float(r["GB"]) / 1e3 / t if t else 0.0
            else:
                r["v"] = float(r[key])
        if a.metric in ROOF:
            T = sum(float(r["t_ms"]) for r in rows)
            rows = [r for r in rows if 100 * float(r["t_ms"]) / T >= a.min_share]
            rows.sort(key=lambda r: -float(r["t_ms"]))
            plot_abs(rows, a, path, T)
            continue
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
 