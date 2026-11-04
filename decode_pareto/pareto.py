#!/usr/bin/env python3
"""
pareto.py -- plots and Pareto analysis for decode-phase DVFS sweeps.

Reads decode_metrics.csv (from extract.py) and emits:
  fig1_energy_vs_knob.png   J/token against power cap and against SM clock
  fig2_time_vs_knob.png     ms/token against the same two knobs
  fig3_pareto.png           latency-energy trade-off, one panel per batch size
  fig4_pareto_4b_vs_8b.png  the same trade-off across model size at BS=1
  pareto_front.csv          which configurations survive domination

A point dominates another when it is no worse on BOTH axes and strictly
better on at least one. Survivors form the Pareto front; everything else is a
configuration you would never deliberately choose.

Axes are log-scaled: the 210 MHz points are several times slower than the
rest and would otherwise flatten every other point into the corner.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = list(csv.DictReader(open("decode_metrics.csv")))
for r in ROWS:
    for k in ("bs", "setting", "sm_mhz"):
        r[k] = int(r[k])
    for k in ("j_per_tok", "ms_per_tok", "tok_per_s", "decode_w", "cap_pct"):
        r[k] = float(r[k])

BSS = sorted({r["bs"] for r in ROWS if r["model"] == "qwen3_4b"})
CMAP = plt.get_cmap("viridis")
COL = {b: CMAP(i / max(1, len(BSS) - 1)) for i, b in enumerate(BSS)}
CAP_C, CLK_C = "#c0504d", "#1f77b4"

def sel(bs, mode, model="qwen3_4b", drop_210=False):
    rs = [r for r in ROWS if r["model"] == model
          and r["bs"] == bs and r["mode"] == mode]
    if drop_210:
        # 210 MHz is dominated everywhere and 3-6x slower than any other
        # point, so plotting it compresses the interesting region to nothing.
        rs = [r for r in rs if r["setting"] != 210]
    return sorted(rs, key=lambda r: r["setting"])

def knob_figure(ykey, ylabel, fname, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for bs in BSS:
        cp, ck = sel(bs, "cap"), sel(bs, "clock")
        ax[0].plot([r["decode_w"] for r in cp], [r[ykey] for r in cp],
                   "o-", color=COL[bs], label=f"BS={bs}", ms=5)
        ax[1].plot([r["sm_mhz"] for r in ck], [r[ykey] for r in ck],
                   "s-", color=COL[bs], label=f"BS={bs}", ms=5)
    ax[0].set_xlabel("measured decode power (W)")
    ax[0].set_title("power cap sweep")
    ax[1].set_xlabel("achieved SM clock (MHz)")
    ax[1].set_title("SM clock sweep")
    for a in ax:
        a.set_ylabel(ylabel)
        a.set_yscale("log")
        a.grid(alpha=.3, which="both")
        a.legend(fontsize=8)
    fig.suptitle(f"{title}   Qwen3-4B, A100 80GB PCIe, seq=1024, decode phase only")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print("wrote", fname)

knob_figure("j_per_tok", "J / token", "fig1_energy_vs_knob.png", "Energy per token")
knob_figure("ms_per_tok", "ms / token", "fig2_time_vs_knob.png", "Latency per token")

# Repeat-to-repeat spread of J/token within a single configuration runs up to
# 3.3% in this dataset (median 0.7%), so differences below that are noise.
# A strict dominance test would rank configurations that are statistically
# indistinguishable, which is how cap 225, clock 810 and clock 1110 at BS=32
# ended up listed as three separate front points 0.3% apart.
TOL = 0.03

def pareto(points, tol=TOL):
    """Dominance with a tolerance band.

    q dominates p only if q is better than p by more than `tol` on at least one
    axis and not worse by more than `tol` on the other. Points that merely tie
    within the band both survive, and are reported as a tie group."""
    out = []
    for p in points:
        if not any(q is not p
                   and q["ms_per_tok"] <= p["ms_per_tok"] * (1 + tol)
                   and q["j_per_tok"] <= p["j_per_tok"] * (1 + tol)
                   and (q["ms_per_tok"] < p["ms_per_tok"] * (1 - tol)
                        or q["j_per_tok"] < p["j_per_tok"] * (1 - tol))
                   for q in points):
            out.append(p)
    return sorted(out, key=lambda r: r["ms_per_tok"])

def tie_groups(front, tol=TOL):
    """Cluster front points whose J/token agree to within tol."""
    groups = []
    for p in sorted(front, key=lambda r: r["j_per_tok"]):
        if groups and p["j_per_tok"] <= groups[-1][0]["j_per_tok"] * (1 + tol):
            groups[-1].append(p)
        else:
            groups.append([p])
    return groups

def panel(ax, pts, title):

    fr = pareto(pts)
    for mode, mk, col in (("cap", "o", CAP_C), ("clock", "s", CLK_C)):
        g = [r for r in pts if r["mode"] == mode]
        ax.scatter([r["ms_per_tok"] for r in g], [r["j_per_tok"] for r in g],
                   marker=mk, s=80, color=col, label=mode, zorder=3,
                   edgecolor="white", linewidth=.9)
    ax.plot([r["ms_per_tok"] for r in fr], [r["j_per_tok"] for r in fr],
            "--", color="0.45", lw=1.3, zorder=2, label="Pareto front")
    # Shade +/- TOL around the cheapest point: anything inside is within the
    # repeat-to-repeat noise of the best configuration and cannot be ranked.
    jmin = min(r["j_per_tok"] for r in pts)
    ax.axhspan(jmin * (1 - TOL), jmin * (1 + TOL), color="0.85", zorder=0,
               label=f"+/-{TOL:.0%} noise band")
    for r in pts:
        on = r in fr
        ax.annotate(f"{r['setting']}", (r["ms_per_tok"], r["j_per_tok"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5,
                    color="black" if on else "0.55",
                    fontweight="bold" if on else "normal")
    ax.set_title(title)
    ax.set_xlabel("ms / token")
    ax.grid(alpha=.3, which="both")
    return fr

fig, axes = plt.subplots(1, len(BSS), figsize=(4.1 * len(BSS), 4.4))
front_rows = []
for ax, bs in zip(axes, BSS):
    pts = sel(bs, "cap", drop_210=True) + sel(bs, "clock", drop_210=True)
    fr = panel(ax, pts, f"BS = {bs}")
    front_rows += [dict(p, on_front=int(p in fr)) for p in pts]
    if bs == BSS[0]:
        ax.set_ylabel("J / token"); ax.legend(fontsize=8, loc="upper right")
fig.suptitle("Decode Pareto front, latency vs energy. Lower-left is better. "
             "Labels: cap in W (circles) or locked SM clock in MHz (squares). "
             "Bold = on the front. The 210 MHz point is omitted: dominated everywhere.", fontsize=10)
fig.tight_layout()
fig.savefig("fig3_pareto.png", dpi=150)
print("wrote fig3_pareto.png")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
for ax, mdl in zip(axes, ("qwen3_4b", "qwen3_8b")):
    pts = sel(1, "cap", mdl, drop_210=True) + sel(1, "clock", mdl, drop_210=True)
    panel(ax, pts, f"{mdl}, BS=1")
    if mdl == "qwen3_4b":
        ax.set_ylabel("J / token"); ax.legend(fontsize=8)
fig.suptitle("Model size at BS=1: the same trade-off shifts up and right", fontsize=11)
fig.tight_layout()
fig.savefig("fig4_pareto_4b_vs_8b.png", dpi=150)
print("wrote fig4_pareto_4b_vs_8b.png")

keys = ["model", "bs", "mode", "setting", "ms_per_tok", "j_per_tok",
        "tok_per_s", "decode_w", "sm_mhz", "cap_pct", "on_front"]
with open("pareto_front.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(sorted(front_rows, key=lambda r: (r["bs"], r["ms_per_tok"])))
print("wrote pareto_front.csv")

print("\n=== Pareto front, fastest to cheapest ===")
for bs in BSS:
    fr = sorted((r for r in front_rows if r["bs"] == bs and r["on_front"]),
                key=lambda r: r["ms_per_tok"])
    nc = sum(1 for r in fr if r["mode"] == "cap")
    print(f"\nBS={bs}  ({nc} cap, {len(fr)-nc} clock on front, tol={TOL:.0%})")
    for g in tie_groups(fr):
        tag = "  <- tie, indistinguishable" if len(g) > 1 else ""
        for r in sorted(g, key=lambda x: x["ms_per_tok"]):
            print(f"   {r['mode']:5} {r['setting']:>5} | {r['ms_per_tok']:>7.3f} ms/tok"
                  f" | {r['j_per_tok']:>7.3f} J/tok | {r['decode_w']:>6.1f} W"
                  f" | SM {r['sm_mhz']:>4} | cap {r['cap_pct']:>5.1f}%{tag}")
            tag = ""
        if len(g) > 1:
            print(f"         {'':>5} | {'':>7}         | "
                  f"{'':>7}         | ^ same operating point reached two ways")
