#!/usr/bin/env python3
"""Bytes per decode step from a dmon file and the client output.

usage: net_per_step.py CLIENT_LOG DMON_FILE [--bg-tx 10] [--bg-rx 30]

CLIENT_LOG  output of bench_decode_offline.py (MARK and WORKLOAD_JSON lines)
DMON_FILE   output of: nvidia-smi dmon -s t -d 1 -o T
            columns: time gpu rxpci txpci   (MB/s)
"""
import argparse
import json
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("client_log")
    ap.add_argument("dmon_file")
    ap.add_argument("--bg-tx", type=float, default=0.0, help="MB/s to subtract")
    ap.add_argument("--bg-rx", type=float, default=0.0, help="MB/s to subtract")
    a = ap.parse_args()

    marks, info = {}, None
    for line in open(a.client_log):
        if line.startswith("MARK "):
            _, name, t = line.split()
            marks[name] = float(t)
        elif line.startswith(("DECODE_JSON ", "WORKLOAD_JSON ")):
            info = json.loads(line.split(" ", 1)[1])
    lo, hi = marks["decode_start"], marks["decode_end"]
    steps = info["steps"]
    day = time.strftime("%Y-%m-%d", time.localtime(lo))

    rows = {}
    for line in open(a.dmon_file):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 4:
            continue
        t = time.mktime(time.strptime(day + " " + p[0], "%Y-%m-%d %H:%M:%S"))
        if lo <= t <= hi:
            rows.setdefault(p[1], []).append((int(p[2]), int(p[3])))

    print(f"decode window {hi - lo:.2f} s, {steps} steps "
          f"({info['step_ms']} ms per step), bs={info['bs']}")
    for g in sorted(rows):
        v = rows[g]
        n = len(v)
        rx = sum(x for x, _ in v) / n
        tx = sum(y for _, y in v) / n
        spread_tx = (max(y for _, y in v) - min(y for _, y in v))
        rx_net, tx_net = rx - a.bg_rx, tx - a.bg_tx
        per_step_tx = tx_net * (hi - lo) / steps
        per_step_rx = rx_net * (hi - lo) / steps
        print(f"GPU {g}: {n} samples  rx {rx:.0f}  tx {tx:.0f} MB/s "
              f"(tx min-max spread {spread_tx} MB/s)  ->  per step: "
              f"tx {per_step_tx:.3f} MB  rx {per_step_rx:.3f} MB")
    if not rows:
        print("no dmon samples inside the decode window; "
              "check that dmon was running and both clocks agree")


if __name__ == "__main__":
    main()