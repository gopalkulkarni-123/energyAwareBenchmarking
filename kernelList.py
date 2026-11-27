#!/usr/bin/env python3
"""List every kernel launch in one profiled decode step (ncu raw CSV, base units).

usage: python3 kernel_list.py results/roofline/csv/ncu_bs1.csv [--layers 36] [--seq 80]

Prints:
  1. per unique kernel: group, launches, launches per layer, time, bytes, FLOPs
  2. the first --seq launches in launch order, to see the sequence inside one layer
Writes kernel_list_<file>.csv (per unique kernel) and kernel_seq_<file>.csv (every launch).
Groups use the same rules as rooflineSummary.py.
"""
import argparse
import csv
import os

M = {
    "t": "gpu__time_duration.sum",
    "rd": "dram__bytes_read.sum",
    "wr": "dram__bytes_write.sum",
    "tensor": "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "fadd": "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "fmul": "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "ffma": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "hadd": "sm__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "hmul": "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "hfma": "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
}
GROUPS = [
    ("attention", ("attn", "attention", "flash", "paged", "unified")),
    ("matmul", ("gemm", "cutlass", "xmma", "gemv", "splitk", "matmul")),
]


def num(s):
    try:
        return float((s or "").replace(",", "").strip())
    except ValueError:
        return 0.0


def group(name):
    n = name.lower()
    for g, keys in GROUPS:
        if any(k in n for k in keys):
            return g
    return "other"


def short(name, n=70):
    # drop template arguments and argument lists: keep the readable head
    s = name.split("(")[0]
    depth, out = 0, []
    for ch in s:
        if ch == "<":
            depth += 1
            if depth == 1:
                out.append("<...>")
            continue
        if ch == ">":
            depth -= 1
            continue
        if depth == 0:
            out.append(ch)
    s = "".join(out).replace("void ", "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--seq", type=int, default=80, help="launches to print in order")
    a = ap.parse_args()

    rows = list(csv.reader(open(a.csv, newline="")))
    head, data = rows[0], rows[2:]
    col = {h: i for i, h in enumerate(head)}
    kn = col["Kernel Name"]
    launches = []
    for i, r in enumerate(data):
        if len(r) < len(head):
            continue
        v = {k: (num(r[col[m]]) if m in col else 0.0) for k, m in M.items()}
        f_other = v["fadd"] + v["fmul"] + 2 * v["ffma"] + v["hadd"] + v["hmul"] + 2 * v["hfma"]
        launches.append(dict(i=i, name=r[kn], short=short(r[kn]), group=group(r[kn]),
                             t=v["t"] * 1e-9, B=v["rd"] + v["wr"],
                             F=v["tensor"] + f_other))

    T = sum(k["t"] for k in launches)
    per = {}
    for k in launches:
        p = per.setdefault(k["name"], dict(name=k["name"], short=k["short"], group=k["group"], n=0,
                                           t=0.0, B=0.0, F=0.0, first=k["i"]))
        p["n"] += 1
        p["t"] += k["t"]
        p["B"] += k["B"]
        p["F"] += k["F"]

    print(f"{os.path.basename(a.csv)}: {len(launches)} launches, {T*1e3:.2f} ms kernel time\n")
    print(f"{'group':<10}{'launches':>9}{'/layer':>7}{'ms':>8}{'% time':>7}{'GB':>8}"
          f"{'TB/s':>6}{'GFLOP':>8}  kernel")
    out = []
    for g in ("matmul", "attention", "other"):
        sel = sorted([p for p in per.values() if p["group"] == g], key=lambda p: -p["t"])
        for p in sel:
            bw = p["B"] / p["t"] / 1e12 if p["t"] else 0
            print(f"{g:<10}{p['n']:9d}{p['n']/a.layers:7.2f}{p['t']*1e3:8.3f}"
                  f"{100*p['t']/T:6.1f}%{p['B']/1e9:8.3f}{bw:6.2f}{p['F']/1e9:8.2f}  {p['short']}")
            out.append(dict(group=g, launches=p["n"], per_layer=round(p["n"] / a.layers, 2),
                            t_ms=round(p["t"] * 1e3, 4), pct_time=round(100 * p["t"] / T, 2),
                            GB=round(p["B"] / 1e9, 4), TBps=round(bw, 3),
                            GFLOP=round(p["F"] / 1e9, 3), short=p["short"], name=p["name"]))
        n = sum(p["n"] for p in sel)
        t = sum(p["t"] for p in sel)
        print(f"{'':<10}{n:9d}{n/a.layers:7.2f}{t*1e3:8.3f}{100*t/T:6.1f}%  <- {g} total\n")

    print(f"first {a.seq} launches, in launch order:")
    for k in launches[: a.seq]:
        print(f"  {k['i']:4d}  {k['group']:<10}{k['t']*1e6:8.1f} us  {k['short']}")

    base = os.path.splitext(os.path.basename(a.csv))[0]
    with open(f"kernel_list_{base}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    with open(f"kernel_seq_{base}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["launch", "group", "t_us", "MB", "MFLOP", "short", "name"])
        for k in launches:
            w.writerow([k["i"], k["group"], round(k["t"] * 1e6, 2), round(k["B"] / 1e6, 3),
                        round(k["F"] / 1e6, 3), k["short"], k["name"]])
    print(f"\nwrote kernel_list_{base}.csv and kernel_seq_{base}.csv")


if __name__ == "__main__":
    main()
