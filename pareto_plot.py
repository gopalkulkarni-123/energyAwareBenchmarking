#!/usr/bin/env python3
"""Pareto plot: throughput (tokens/s) vs energy efficiency (tokens/J).

usage: python3 pareto_tput.py [results/dvfs/decode_metrics.csv] [--out results/dvfs]

One panel per batch size. Both axes: higher is better, so the front is the
upper-right edge. A point is dominated if another point is better by more
than TOL on one axis and not worse by more than TOL on the other.
Writes pareto_tput_tokj.png and pareto_tput_front.csv.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOL = 0.03          # repeat-to-repeat spread of J/token, about 3%
STYLE = {"clock": ("^", "#1baf7a", "clock lock (MHz)"),
         "cap": ("s", "#eb6834", "power cap (W)"),
         "base": ("o", "#2a78d6", "base (no limit)")}


def front(pts):
    out = []
    for p in pts:
        if not any(q is not p
                   and q["tps"] >= p["tps"] * (1 - TOL)
                   and q["tpj"] >= p["tpj"] * (1 - TOL)
                   and (q["tps"] > p["tps"] * (1 + TOL)
                        or q["tpj"] > p["tpj"] * (1 + TOL))
                   for q in pts):
            out.append(p)
    return sorted(out, key=lambda r: r["tps"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="?", default="results/dvfs/decode_metrics.csv")
    ap.add_argument("--out", default="results/dvfs")
    a = ap.parse_args()

    rows = []
    for r in csv.DictReader(open(a.metrics)):
        rows.append(dict(bs=int(r["bs"]), mode=r["mode"], setting=float(r["setting"]),
                         sm_mhz=float(r["sm_mhz"]), tps=float(r["tok_per_s"]),
                         tpj=1.0 / float(r["j_per_tok"]), w=float(r["decode_w"])))
    bss = sorted({r["bs"] for r in rows})

    ink, muted, grid = "#2b2b29", "#6b6b66", "#e4e3df"
    fig, axes = plt.subplots(1, len(bss), figsize=(4.6 * len(bss), 4.6))
    if len(bss) == 1:
        axes = [axes]
    out_rows = []
    print(f"Pareto front per batch (tolerance {TOL:.0%}), fastest first")
    for ax, bs in zip(axes, bss):
        pts = [r for r in rows if r["bs"] == bs]
        fr = front(pts)
        for mode, (mk, c, lab) in STYLE.items():
            g = [r for r in pts if r["mode"] == mode]
            if g:
                ax.scatter([r["tps"] for r in g], [r["tpj"] for r in g], marker=mk,
                           s=70, color=c, edgecolor="white", linewidth=0.9,
                           zorder=3, label=lab)
        ax.plot([r["tps"] for r in fr], [r["tpj"] for r in fr], "--", color=muted,
                linewidth=1.3, zorder=2)
        for r in pts:
            on = r in fr
            ax.annotate(f"{r['setting']:.0f}", (r["tps"], r["tpj"]), xytext=(5, 4),
                        textcoords="offset points", fontsize=8,
                        color=ink if on else muted,
                        fontweight="bold" if on else "normal")
        ax.set_title(f"batch {bs}", color=ink, loc="left", fontsize=11)
        ax.set_xlabel("Throughput (tokens/s)", color=ink)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(muted)
        ax.tick_params(colors=muted)
        print(f"\nbatch {bs}")
        for r in fr:
            unit = "MHz" if r["mode"] == "clock" else "W"
            print(f"  {r['mode']:<5} {r['setting']:6.0f} {unit:<3} "
                  f"{r['tps']:9.1f} tok/s  {r['tpj']:7.3f} tok/J  "
                  f"{r['w']:6.1f} W  SM {r['sm_mhz']:.0f} MHz")
        out_rows += [dict(bs=bs, mode=r["mode"], setting=r["setting"],
                          sm_mhz=r["sm_mhz"], tok_per_s=round(r["tps"], 2),
                          tok_per_j=round(r["tpj"], 4), decode_w=r["w"],
                          on_front=int(r in fr)) for r in pts]
    axes[0].set_ylabel("Energy efficiency (tokens/J)", color=ink)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False)
    fig.suptitle("Qwen3-4B decode, A100 80GB PCIe: throughput vs tokens per joule. "
                 "Upper right is better. Bold labels and dashed line: Pareto front.",
                 color=ink, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "pareto_tput_tokj.png"), dpi=150)
    with open(os.path.join(a.out, "pareto_tput_front.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote pareto_tput_tokj.png and pareto_tput_front.csv in {a.out}")


if __name__ == "__main__":
    main()
