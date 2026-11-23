#!/usr/bin/env python3
"""Energy tables and plots from decode_metrics.csv (decode_pareto/extract.py).

usage: python3 energy_tables.py [results/dvfs/decode_metrics.csv] [--out results/dvfs]

Writes:
  energy_table.csv     one row per batch size and setting, with step time and
                       changes against the 1,410 MHz clock lock of that batch
  energy_summary.csv   per batch size: the setting with the lowest J per token
  step_time_vs_clock.png, j_per_token_vs_clock.png   (clock locks only)
"""
import argparse
import csv
import os

COLS = ["bs", "mode", "setting", "sm_mhz", "step_ms", "slower_pct",
        "decode_w", "j_per_tok", "energy_saved_pct", "cap_pct", "reps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="?", default="results/dvfs/decode_metrics.csv")
    ap.add_argument("--out", default="results/dvfs")
    ap.add_argument("--ref-clock", type=float, default=1410)
    a = ap.parse_args()

    rows = []
    for r in csv.DictReader(open(a.metrics)):
        bs = int(r["bs"])
        rows.append(dict(bs=bs, mode=r["mode"], setting=float(r["setting"]),
                         sm_mhz=float(r["sm_mhz"]),
                         step_ms=float(r["ms_per_tok"]) * bs,
                         decode_w=float(r["decode_w"]), j_per_tok=float(r["j_per_tok"]),
                         cap_pct=float(r["cap_pct"]), reps=int(r["reps"])))

    out_rows, summary = [], []
    for bs in sorted({r["bs"] for r in rows}):
        sel = [r for r in rows if r["bs"] == bs]
        ref = [r for r in sel if r["mode"] == "clock" and r["setting"] == a.ref_clock]
        if not ref:
            print(f"batch {bs}: no {a.ref_clock:.0f} MHz clock row, skipped")
            continue
        ref = ref[0]
        sel.sort(key=lambda r: (r["mode"] != "clock", r["setting"]))
        print(f"\nbatch {bs}  (reference: clock {a.ref_clock:.0f} MHz, "
              f"{ref['step_ms']:.2f} ms per step, {ref['j_per_tok']:.4f} J per token)")
        print(f"  {'setting':<16}{'clock':>7}{'step ms':>9}{'slower':>8}"
              f"{'power W':>9}{'J/token':>9}{'saved':>9}{'cap %':>7}")
        for r in sel:
            r["slower_pct"] = 100 * (r["step_ms"] / ref["step_ms"] - 1)
            r["energy_saved_pct"] = 100 * (1 - r["j_per_tok"] / ref["j_per_tok"])
            unit = "MHz" if r["mode"] == "clock" else "W"
            name = f"{r['mode']} {r['setting']:.0f} {unit}"
            print(f"  {name:<16}{r['sm_mhz']:7.0f}{r['step_ms']:9.2f}{r['slower_pct']:7.1f}%"
                  f"{r['decode_w']:9.1f}{r['j_per_tok']:9.4f}{r['energy_saved_pct']:8.1f}%"
                  f"{r['cap_pct']:7.1f}")
            out_rows.append({k: (round(r[k], 4) if isinstance(r[k], float) else r[k])
                             for k in COLS})
        best = min(sel, key=lambda r: r["j_per_tok"])
        summary.append(dict(bs=bs, best_mode=best["mode"], best_setting=best["setting"],
                            best_clock_mhz=best["sm_mhz"],
                            j_per_tok=round(best["j_per_tok"], 4),
                            energy_saved_pct=round(best["energy_saved_pct"], 1),
                            slower_pct=round(best["slower_pct"], 1)))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "energy_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out_rows)
    with open(os.path.join(a.out, "energy_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\nbest setting per batch size (lowest J per token)")
    for s in summary:
        unit = "MHz" if s["best_mode"] == "clock" else "W"
        print(f"  batch {s['bs']:>3}: {s['best_mode']} {s['best_setting']:.0f} {unit} "
              f"(runs at {s['best_clock_mhz']:.0f} MHz), {s['j_per_tok']:.4f} J/token, "
              f"{s['energy_saved_pct']:.1f}% less energy, {s['slower_pct']:.1f}% slower")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: tables written, no plots")
        return
    ink, muted, grid = "#2b2b29", "#6b6b66", "#e4e3df"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    markers = ["o", "s", "^", "D"]
    for key, ylabel, title, fname in (
            ("step_ms", "Step time (ms)", "step time by clock lock",
             "step_time_vs_clock.png"),
            ("rel_energy", "Energy per token, relative to 1,410 MHz",
             "energy per token by clock lock",
             "j_per_token_vs_clock.png")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, bs in enumerate(sorted({r["bs"] for r in out_rows})):
            pts = sorted([r for r in out_rows if r["bs"] == bs and r["mode"] == "clock"],
                         key=lambda r: r["setting"])
            xs = [r["setting"] for r in pts]
            if key == "rel_energy":
                ys = [1 - r["energy_saved_pct"] / 100 for r in pts]
            else:
                ys = [r[key] for r in pts]
            c, m = colors[i % 4], markers[i % 4]
            ax.plot(xs, ys, marker=m, color=c, linewidth=2, markersize=7,
                    label=f"batch {bs}")
            if key == "step_ms":
                ax.annotate(f"batch {bs}", (xs[-1], ys[-1]), xytext=(6, 0),
                            textcoords="offset points", color=ink, fontsize=9, va="center")
        ax.set_xlabel("SM clock lock (MHz)", color=ink)
        ax.set_ylabel(ylabel, color=ink)
        if key == "step_ms":
            ax.set_ylim(bottom=0)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(muted)
        ax.tick_params(colors=muted)
        ax.legend(frameon=False, loc="best")
        ax.set_title(f"Qwen3-4B decode, A100 80GB PCIe: {title}",
                     color=ink, loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, fname), dpi=150)
        plt.close(fig)
    print(f"\nwrote energy_table.csv, energy_summary.csv, step_time_vs_clock.png, "
          f"j_per_token_vs_clock.png in {a.out}")


if __name__ == "__main__":
    main()
