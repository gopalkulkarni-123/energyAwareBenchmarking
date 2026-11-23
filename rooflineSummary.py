#!/usr/bin/env python3
"""Sum an ncu raw CSV (one decode step) into Roofline numbers.

usage: python3 roofline_summary.py ncu_bs1.csv [ncu_bs4.csv ...]

Export each report first, in base units:
  ncu --import ncu_bs1.ncu-rep --csv --page raw --print-units base > ncu_bs1.csv

Per kernel:
  F_tensor = sm__ops_path_tensor_src_bf16_dst_fp32   (counts FLOPs, checked
             in the calibration; includes the padded work of small batches)
  F_other  = fadd + fmul + 2*ffma + hadd + hmul + 2*hfma   (thread instructions;
             a packed half2 instruction does twice the work, so this part is
             a lower bound)
  B        = dram bytes read + written
  t        = gpu__time_duration
Kernels are grouped by name: matmul, attention, other.
"""
import csv
import os
import sys

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
GROUPS = [   # attention first: some attention kernels contain "splitk" in their name
    ("attention", ("attn", "attention", "flash", "paged", "unified")),
    ("matmul", ("gemm", "cutlass", "xmma", "gemv", "splitk", "matmul")),
]


def num(s):
    s = (s or "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def group(name):
    n = name.lower()
    for g, keys in GROUPS:
        if any(k in n for k in keys):
            return g
    return "other"


def read(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    head, units, data = rows[0], rows[1], rows[2:]
    col = {name: i for i, name in enumerate(head)}
    missing = [m for m in M.values() if m not in col]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    for key in ("t", "rd"):
        u = units[col[M[key]]].strip().lower()
        if u not in ("ns", "nsecond", "byte", "bytes", ""):
            print(f"warning: {M[key]} is in '{units[col[M[key]]]}', "
                  f"export with --print-units base")
    kname = col["Kernel Name"]
    out = []
    for r in data:
        if len(r) < len(head):
            continue
        v = {k: num(r[col[m]]) for k, m in M.items()}
        f_other = (v["fadd"] + v["fmul"] + 2 * v["ffma"]
                   + v["hadd"] + v["hmul"] + 2 * v["hfma"])
        out.append(dict(name=r[kname], group=group(r[kname]), t=v["t"] * 1e-9,
                        B=v["rd"] + v["wr"], F_tensor=v["tensor"], F_other=f_other))
    return out


def show(label, ks):
    t = sum(k["t"] for k in ks)
    B = sum(k["B"] for k in ks)
    Ft = sum(k["F_tensor"] for k in ks)
    Fo = sum(k["F_other"] for k in ks)
    F = Ft + Fo
    I = F / B if B else float("nan")
    print(f"  {label:<10}{len(ks):5d} kernels  t {t*1e3:7.3f} ms  B {B/1e9:7.3f} GB  "
          f"F {F/1e9:8.2f} GFLOP (tensor {Ft/1e9:8.2f}, other {Fo/1e9:6.2f})  "
          f"I {I:6.2f} FLOP/B  {F/t/1e12 if t else 0:6.2f} TFLOP/s  "
          f"{B/t/1e12 if t else 0:5.3f} TB/s")
    return dict(kernels=len(ks), t_s=t, B=B, F_tensor=Ft, F_other=Fo, F=F, I=I)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows_out = []
    for path in sys.argv[1:]:
        ks = read(path)
        print(f"{os.path.basename(path)}: one decode step")
        for g in ("matmul", "attention", "other"):
            sel = [k for k in ks if k["group"] == g]
            if sel:
                s = show(g, sel)
                rows_out.append(dict(file=path, group=g, **s))
        s = show("ALL", ks)
        rows_out.append(dict(file=path, group="ALL", **s))
        names = {}
        for k in ks:
            if k["group"] == "other":
                names[k["name"][:60]] = names.get(k["name"][:60], 0) + k["t"]
        top = sorted(names.items(), key=lambda x: -x[1])[:5]
        if top:
            print("  largest 'other' kernels by time:")
            for n, t in top:
                print(f"    {t*1e6:8.1f} us  {n}")
        print()
    with open("roofline_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("wrote roofline_summary.csv")


if __name__ == "__main__":
    main()