#!/usr/bin/env python3
"""
dvfs_probe.py -- capability probe and sweep tool for NVIDIA GPU power/frequency knobs

Runs as a NORMAL USER in the user's own shell environment. Only the three
nvidia-smi calls that change GPU state (-pl, -lgc, -rgc) are elevated, through
`sudo -n nvidia-smi`. Everything else -- NVML sampling and the workload
itself -- runs unprivileged, so the workload sees exactly the environment you
get when you run it by hand.

One-time setup (root):
    sudo visudo -f /etc/sudoers.d/nvidia-smi
  add the line (adjust user and path; `which nvidia-smi` gives the path):
    <user> ALL=(root) NOPASSWD: /usr/bin/nvidia-smi
Verify with:
    sudo -n nvidia-smi -L

Three modes:
  probe   Read-only. Prints this card's power-limit bounds and its supported
          SM/MEM frequency steps, and optionally runs a "requested vs actual
          clock" clamp test (--test-clamp, needs the sudoers rule).
  sweep   Under a given workload, sweeps power cap and SM clock while sampling
          power / clock / throttle reasons. Reports energy, a throughput proxy
          and the cap-engagement rate for every configuration.
  monitor Sampling only, changes nothing. Use it to instrument a job that is
          already running.

Key output: pct_swpowercap = fraction of samples where the SwPowerCap throttle
reason was asserted. This is the direct empirical evidence for whether a power
cap actually engages at all.

Requires: pip install nvidia-ml-py
          (the built-in synthetic workload additionally needs torch)
"""

import argparse
import atexit
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

try:
    import pynvml
except ImportError:
    sys.exit("pynvml missing, install it first: pip install nvidia-ml-py")


# --------------------------------------------------------------------------
# NVML compatibility shim: newer versions renamed ThrottleReason to EventReason
# --------------------------------------------------------------------------

def _const(name, default):
    for prefix in ("nvmlClocksThrottleReason", "nvmlClocksEventReason"):
        if hasattr(pynvml, prefix + name):
            return getattr(pynvml, prefix + name)
    return default


THROTTLE_BITS = [
    ("GpuIdle",                   _const("GpuIdle", 1 << 0)),
    ("AppClocksSetting",          _const("ApplicationsClocksSetting", 1 << 1)),
    ("SwPowerCap",                _const("SwPowerCap", 1 << 2)),
    ("HwSlowdown",                _const("HwSlowdown", 1 << 3)),
    ("SyncBoost",                 _const("SyncBoost", 1 << 4)),
    ("SwThermalSlowdown",         _const("SwThermalSlowdown", 1 << 5)),
    ("HwThermalSlowdown",         _const("HwThermalSlowdown", 1 << 6)),
    ("HwPowerBrakeSlowdown",      _const("HwPowerBrakeSlowdown", 1 << 7)),
    ("DisplayClockSetting",       _const("DisplayClockSetting", 1 << 8)),
]


def _get_throttle(handle):
    for fn in ("nvmlDeviceGetCurrentClocksThrottleReasons",
               "nvmlDeviceGetCurrentClocksEventReasons"):
        if hasattr(pynvml, fn):
            return getattr(pynvml, fn)(handle)
    return 0


def decode_throttle(mask):
    return [name for name, bit in THROTTLE_BITS if mask & bit]


# --------------------------------------------------------------------------
# Sampling thread (read-only NVML, no privileges needed)
# --------------------------------------------------------------------------

class Sampler(threading.Thread):
    """Collects power / clock / temp / util / throttle at a fixed interval."""

    def __init__(self, handle, interval_ms=50):
        super().__init__(daemon=True)
        self.h = handle
        self.dt = interval_ms / 1000.0
        self.rows = []
        # NOTE: must not be called _stop -- threading.Thread already defines a
        # private _stop() method, and Thread.join() calls it internally.
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            t = time.time()
            try:
                row = dict(
                    t=t,
                    power_w=pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0,
                    sm_mhz=pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),
                    mem_mhz=pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_MEM),
                    temp_c=pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU),
                    util_pct=pynvml.nvmlDeviceGetUtilizationRates(self.h).gpu,
                    throttle=_get_throttle(self.h),
                )
                self.rows.append(row)
            except pynvml.NVMLError:
                pass
            time.sleep(max(0.0, self.dt - (time.time() - t)))

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)

    # ---- derived quantities ----

    def slice(self, t0=None, t1=None):
        """Return the samples inside the [t0, t1] window, extended by one extra
        sample on each side so the trapezoidal integral spans the whole interval
        rather than starting at the first strictly-interior sample."""
        if t0 is None and t1 is None:
            return list(self.rows)
        lo = 0
        hi = len(self.rows)
        if t0 is not None:
            while lo < hi and self.rows[lo]["t"] < t0:
                lo += 1
            lo = max(0, lo - 1)
        if t1 is not None:
            hi = lo
            while hi < len(self.rows) and self.rows[hi]["t"] <= t1:
                hi += 1
            hi = min(len(self.rows), hi + 1)
        return self.rows[lo:hi]

    @staticmethod
    def energy_trapezoid_j(rows):
        """Trapezoidal integration over the power samples, in joules."""
        if len(rows) < 2:
            return float("nan")
        e = 0.0
        for a, b in zip(rows, rows[1:]):
            e += 0.5 * (a["power_w"] + b["power_w"]) * (b["t"] - a["t"])
        return e

    def summary(self, t0=None, t1=None):
        rows = self.slice(t0, t1)
        if not rows:
            return {}
        pw = [r["power_w"] for r in rows]
        sm = [r["sm_mhz"] for r in rows]
        n = len(rows)
        out = dict(
            n_samples=n,
            window_s=round(rows[-1]["t"] - rows[0]["t"], 3),
            power_median_w=round(statistics.median(pw), 1),
            power_max_w=round(max(pw), 1),
            sm_median_mhz=int(statistics.median(sm)),
            sm_max_mhz=max(sm),
            mem_median_mhz=int(statistics.median(r["mem_mhz"] for r in rows)),
            temp_max_c=max(r["temp_c"] for r in rows),
            util_median_pct=int(statistics.median(r["util_pct"] for r in rows)),
            energy_trapz_j=round(self.energy_trapezoid_j(rows), 2),
        )
        for name, bit in THROTTLE_BITS:
            hits = sum(1 for r in rows if r["throttle"] & bit)
            out["pct_" + name.lower()] = round(100.0 * hits / n, 1)
        return out


# --------------------------------------------------------------------------
# Knob setting via nvidia-smi (the only privileged part)
# --------------------------------------------------------------------------

class PrivilegeError(RuntimeError):
    pass


def _nvidia_smi_path():
    p = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
    return os.path.realpath(p)


def _priv_prefix():
    """[] when already root, otherwise a non-interactive sudo. -n means
    "never prompt": if the sudoers rule is missing we fail immediately with a
    clear message instead of hanging on a password prompt mid-sweep."""
    return [] if os.geteuid() == 0 else ["sudo", "-n"]


def check_privileges(gpu):
    """Verify we can run nvidia-smi with elevated rights before touching
    anything, and print the exact sudoers line if we cannot."""
    smi = _nvidia_smi_path()
    cmd = _priv_prefix() + [smi, "-i", str(gpu), "-q", "-d", "POWER"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "<user>"
    raise PrivilegeError(
        "cannot run nvidia-smi with elevated rights (sudo -n failed).\n"
        "One-time setup, as root:\n"
        "    sudo visudo -f /etc/sudoers.d/nvidia-smi\n"
        "and add this single line:\n"
        f"    {user} ALL=(root) NOPASSWD: {smi}\n"
        "Then verify with:\n"
        f"    sudo -n {smi} -L\n"
        f"sudo said: {r.stderr.strip() or r.stdout.strip()}")


_ACTIVE_KNOBS = []


class Knobs:
    """Wraps the three state-changing nvidia-smi calls. Reads go through
    NVML (unprivileged) so every write is verified by an independent path."""

    def __init__(self, handle, gpu):
        self.h = handle
        self.gpu = str(gpu)
        self.smi = _nvidia_smi_path()
        self.default_limit_w = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000.0
        self._cap_touched = False
        self._clk_touched = False
        _ACTIVE_KNOBS.append(self)
        # This works at exit because nvidia-smi is a separate process and does
        # not care whether our own NVML handle is still open.
        atexit.register(self.reset_all)

    def _run(self, *args):
        cmd = _priv_prefix() + [self.smi, "-i", self.gpu] + [str(a) for a in args]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed: "
                               f"{r.stderr.strip() or r.stdout.strip()}")
        return r.stdout

    def current_cap_w(self):
        return pynvml.nvmlDeviceGetPowerManagementLimit(self.h) / 1000.0

    def set_cap_w(self, watts):
        self._run("-pl", int(watts))
        self._cap_touched = True
        got = self.current_cap_w()
        if abs(got - watts) > 1.0:
            print(f"[warn] requested cap {watts} W but readback reports {got:.0f} W",
                  file=sys.stderr)
        return got

    def set_locked_clock(self, mhz):
        self._run("-lgc", f"{int(mhz)},{int(mhz)}")
        self._clk_touched = True

    def reset_all(self):
        if not (self._cap_touched or self._clk_touched):
            return
        failed = []
        # Order matters: unlock the clock first, then restore the power limit,
        # so we never leave the card stuck in a low-frequency intermediate state.
        if self._clk_touched:
            try:
                self._run("-rgc")
                self._clk_touched = False
            except RuntimeError as e:
                failed.append(f"clock reset: {e}")
        if self._cap_touched:
            try:
                self._run("-pl", int(self.default_limit_w))
                self._cap_touched = False
            except RuntimeError as e:
                failed.append(f"power-limit reset: {e}")
        if failed:
            for f in failed:
                print(f"[warn] {f}", file=sys.stderr)
            print(f"[warn] THE GPU MAY STILL BE MODIFIED. Recover manually with:\n"
                  f"         sudo nvidia-smi -i {self.gpu} -rgc\n"
                  f"         sudo nvidia-smi -i {self.gpu} -pl {self.default_limit_w:.0f}",
                  file=sys.stderr)


# --------------------------------------------------------------------------
# Workload
# --------------------------------------------------------------------------

SYNTHETIC_SRC = """
import time, torch
n = {n}
a = torch.randn(n, device='cuda', dtype=torch.float32)
b = torch.randn(n, device='cuda', dtype=torch.float32)
c = torch.empty_like(a)
torch.cuda.synchronize()
# warmup
for _ in range(50):
    torch.add(a, b, alpha=2.0, out=c)
torch.cuda.synchronize()
iters = 0
t_end = time.time() + {dur}
while time.time() < t_end:
    for _ in range(50):
        torch.add(a, b, alpha=2.0, out=c)
    torch.cuda.synchronize()
    iters += 50
# each add reads 2 arrays and writes 1, all float32
gb = iters * n * 4 * 3 / 1e9
print("WORKLOAD_METRIC %.3f" % gb)   # effective HBM traffic (GB) as throughput proxy
"""


def make_synthetic(duration_s, gib=1.0):
    n = int(gib * (1 << 30) / 4)
    src = SYNTHETIC_SRC.format(n=n, dur=duration_s)
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    f.write(src)
    f.close()
    return [sys.executable, f.name]


def run_workload(cmd, fail_log=None):
    """Run the workload once and return (wall_s, metric, marks, extra, returncode).

    The workload inherits this process's environment unchanged. It may emit
    three kinds of agreed-upon lines on stdout:
      WORKLOAD_METRIC <float>   throughput proxy; nan if absent
      WORKLOAD_JSON   {...}     arbitrary extra fields, merged into result rows
      MARK <name> <epoch>       measurement window boundary. A matching
                                <x>_start / <x>_end pair is cut out as its own
                                window and summarised separately.
    """
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    metric = float("nan")
    marks = {}
    extra = {}
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("WORKLOAD_METRIC"):
            try:
                metric = float(line.split()[1])
            except (IndexError, ValueError):
                pass
        elif line.startswith("WORKLOAD_JSON"):
            try:
                extra.update(json.loads(line[len("WORKLOAD_JSON"):].strip()))
            except (ValueError, json.JSONDecodeError):
                pass
        elif line.startswith("MARK "):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    marks[parts[1]] = float(parts[2])
                except ValueError:
                    pass
    if p.returncode != 0:
        print(f"[warn] workload exit code {p.returncode}", file=sys.stderr)
        # Unique name per run: a fixed filename makes it far too easy to read a
        # stale log from an earlier attempt and chase an error already fixed.
        path = fail_log or time.strftime("workload_fail_%H%M%S.log")
        try:
            with open(path, "w") as f:
                f.write("=== command ===\n" + " ".join(cmd) +
                        "\n\n=== stdout ===\n" + p.stdout +
                        "\n\n=== stderr ===\n" + p.stderr)
            print(f"[warn] full workload output written to {path}", file=sys.stderr)
        except OSError:
            print(p.stderr[-4000:], file=sys.stderr)
    return wall, metric, marks, extra, p.returncode


def paired_windows(marks):
    """Turn {'a_start': t0, 'a_end': t1, ...} into [('a', t0, t1), ...]"""
    out = []
    for k, v in marks.items():
        if k.endswith("_start"):
            base = k[: -len("_start")]
            end = marks.get(base + "_end")
            if end is not None and end > v:
                out.append((base, v, end))
    return sorted(out, key=lambda x: x[1])


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

def cmd_probe(args, h):
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()
    lo_mw, hi_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
    default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h)
    cur_mw = pynvml.nvmlDeviceGetPowerManagementLimit(h)

    print(f"Device          : {name}")
    print(f"Driver          : {pynvml.nvmlSystemGetDriverVersion()}")
    print()
    print("--- Power domain ---")
    print(f"  default limit : {default_mw/1000:.0f} W   (i.e. TDP)")
    print(f"  current limit : {cur_mw/1000:.0f} W")
    print(f"  settable range: {lo_mw/1000:.0f} W .. {hi_mw/1000:.0f} W")
    print(f"  => cap headroom below TDP: {(default_mw - lo_mw)/1000:.0f} W "
          f"({100*(default_mw-lo_mw)/default_mw:.0f}% of TDP)")

    print()
    print("--- Frequency domain ---")
    mems = pynvml.nvmlDeviceGetSupportedMemoryClocks(h)
    print(f"  supported MEM steps: {mems}  "
          f"({'single step, -lmc is a no-op' if len(mems) <= 1 else 'multiple steps, -lmc may work'})")
    gfx = sorted(pynvml.nvmlDeviceGetSupportedGraphicsClocks(h, mems[0]))
    print(f"  supported SM steps : {len(gfx)}, {gfx[0]} .. {gfx[-1]} MHz")
    try:
        base = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        mx = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
        print(f"  current / max SM   : {base} / {mx} MHz")
    except pynvml.NVMLError:
        pass

    print()
    print("--- Privileges ---")
    try:
        check_privileges(args.gpu)
        print(f"  sudo -n {_nvidia_smi_path()}: OK")
    except PrivilegeError as e:
        print(f"  {e}")
        if args.test_clamp:
            return

    if args.test_clamp:
        print()
        print("--- Clamp test (requested vs actual) ---")
        knobs = Knobs(h, args.gpu)
        targets = [gfx[-1], gfx[len(gfx) // 2], gfx[0]]
        wl = make_synthetic(args.clamp_load_s, args.tensor_gib)
        try:
            for want in targets:
                try:
                    knobs.set_locked_clock(want)
                except RuntimeError as e:
                    print(f"  {want:>5} MHz -> set failed: {e}")
                    continue
                s = Sampler(h, args.interval_ms)
                s.start()
                run_workload(wl)
                s.stop()
                got = s.summary().get("sm_median_mhz", -1)
                flag = "" if got == want else f"  <-- clamped to {got}"
                print(f"  {want:>5} MHz -> measured {got:>5} MHz{flag}")
        finally:
            knobs.reset_all()


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------

def cmd_sweep(args, h):
    lo_mw, hi_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
    mems = pynvml.nvmlDeviceGetSupportedMemoryClocks(h)
    gfx = sorted(pynvml.nvmlDeviceGetSupportedGraphicsClocks(h, mems[0]))

    if args.caps:
        caps = [int(x) for x in args.caps.split(",")]
    else:
        lo, hi = lo_mw / 1000, hi_mw / 1000
        caps = [round(lo + (hi - lo) * i / 4) for i in range(5)]
    if args.clocks:
        clocks = [int(x) for x in args.clocks.split(",")]
    else:
        clocks = [gfx[0]] + [gfx[int(len(gfx) * f)] for f in (0.25, 0.5, 0.75)] + [gfx[-1]]
        clocks = sorted(set(clocks))

    if args.workload:
        wl = args.workload.split()
    else:
        wl = make_synthetic(args.load_s, args.tensor_gib)

    print(f"cap sweep points  : {caps} W")
    print(f"clock sweep points: {clocks} MHz")
    print(f"reps per point    : {args.reps}")
    print(f"workload runs as  : uid {os.geteuid()} ({os.environ.get('USER', '?')}), "
          f"environment inherited unchanged")
    print()

    if args.preflight_only:
        # Debug path: run the workload once with stdout/stderr inherited so any
        # traceback lands straight in the terminal. Nothing is captured, nothing
        # is parsed, and no knob is touched.
        print("=== Preflight only (output streamed, GPU untouched) ===")
        print("  " + " ".join(wl))
        print()
        rc = subprocess.run(wl).returncode
        print(f"\n=== workload exit code {rc} ===")
        sys.exit(rc)

    # Privilege check happens BEFORE the (long) preflight so a missing sudoers
    # rule is reported in seconds, not minutes.
    check_privileges(args.gpu)

    results = []

    if not args.no_preflight:
        # Run the workload once at stock settings before touching any knob. A
        # broken workload then fails while the GPU is still untouched, instead
        # of after the first cap has already been applied.
        print("=== Preflight (stock settings, nothing modified yet) ===")
        _, _, pf_marks, _, pf_rc = run_workload(wl)
        if pf_rc != 0:
            print(f"[abort] preflight failed (exit {pf_rc}). "
                  f"The GPU was not modified. Fix the workload first.",
                  file=sys.stderr)
            sys.exit(1)
        pf_win = paired_windows(pf_marks)
        if not pf_win:
            print("[warn] preflight produced no MARK lines; phase separation "
                  "will not work and every row will cover the whole subprocess",
                  file=sys.stderr)
        else:
            print("  windows: " + ", ".join(f"{n} {t1-t0:.2f}s" for n, t0, t1 in pf_win))
        print()

    knobs = Knobs(h, args.gpu)

    class WorkloadFailed(RuntimeError):
        pass

    def one_run(mode, setting):
        s = Sampler(h, args.interval_ms)
        try:
            e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        except pynvml.NVMLError:
            e0 = None
        s.start()
        wall, metric, marks, extra, rc = run_workload(wl)
        s.stop()
        if rc != 0:
            # Never keep going after a failed workload: every later row would be
            # measuring a crash, and the verdict printed from those rows would be
            # confidently wrong.
            raise WorkloadFailed(
                f"workload failed (exit {rc}) at {mode}={setting}. "
                f"See the workload_fail_*.log just written. Aborting the sweep.")
        try:
            e_ctr = (pynvml.nvmlDeviceGetTotalEnergyConsumption(h) - e0) / 1000.0 \
                if e0 is not None else float("nan")
        except pynvml.NVMLError:
            e_ctr = float("nan")

        windows = paired_windows(marks)
        if not windows:
            print(f"[warn] {mode}={setting}: workload emitted no MARK lines, "
                  f"so the whole subprocess lifetime (model load included) is "
                  f"being integrated. Phase energies will be meaningless.",
                  file=sys.stderr)
        base = dict(mode=mode, setting=setting, wall_s=round(wall, 3),
                    metric=metric, energy_counter_j=round(e_ctr, 2))
        # Detect a limit that reverted while the workload was running (can happen
        # without persistence mode, when the driver unloads between clients).
        try:
            cap_after = knobs.current_cap_w()
            base["cap_after_w"] = round(cap_after, 1)
            if mode == "cap" and abs(cap_after - setting) > 1.0:
                print(f"[warn] cap was {setting} W before the run but {cap_after:.0f} W "
                      f"after; enable persistence mode (nvidia-smi -pm 1)", file=sys.stderr)
        except pynvml.NVMLError:
            pass
        base.update(extra)

        row = dict(base, phase="whole")
        row.update(s.summary())
        results.append(row)
        if not windows:
            _print_row(mode, setting, row)
        for name, t0, t1 in windows:
            r = dict(base, phase=name)
            r.update(s.summary(t0, t1))
            results.append(r)
            _print_row(mode, setting, r)

    def _print_row(mode, setting, row):
        if not row.get("n_samples"):
            print(f"  [{mode:5}] {setting:>6} {row.get('phase',''):>10} | no samples in window")
            return
        print(f"  [{mode:5}] {setting:>6} {row.get('phase',''):>10} | "
              f"{row['window_s']:>6.2f} s | P_med {row['power_median_w']:>6.1f} W | "
              f"SM_med {row['sm_median_mhz']:>5} MHz | E {row['energy_trapz_j']:>8.1f} J | "
              f"cap {row.get('pct_swpowercap', 0.0):>5.1f}%")

    aborted = None
    try:
        print("=== Power cap sweep (clock left unlocked) ===")
        for cap in caps:
            try:
                knobs.set_cap_w(cap)
            except RuntimeError as e:
                print(f"  [cap  ] {cap:>6} | set failed: {e}")
                continue
            for _ in range(args.reps):
                one_run("cap", cap)
        knobs.reset_all()

        print()
        print("=== SM clock sweep (cap restored to default) ===")
        for clk in clocks:
            try:
                knobs.set_locked_clock(clk)
            except RuntimeError as e:
                print(f"  [clock] {clk:>6} | set failed: {e}")
                continue
            for _ in range(args.reps):
                one_run("clock", clk)
    except WorkloadFailed as e:
        aborted = str(e)
        print(f"\n[abort] {aborted}", file=sys.stderr)
    finally:
        knobs.reset_all()

    if args.out:
        head = ["mode", "setting", "phase", "window_s", "power_median_w",
                "sm_median_mhz", "energy_trapz_j", "pct_swpowercap"]
        rest = sorted({k for r in results for k in r} - set(head))
        keys = [k for k in head if any(k in r for r in results)] + rest
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"\nwrote {args.out}  ({len(results)} rows)")

    if aborted:
        print("\n=== No verdict ===")
        print("  The sweep did not complete, so no conclusion is drawn from "
              "the rows above.")
        sys.exit(1)

    _verdict(results)


def _verdict(results, phase=None):
    """With phase=None, prefer a marker window (usually decode); else 'whole'."""
    phases = {r.get("phase") for r in results}
    if phase is None:
        phase = "whole" if phases == {"whole"} else \
            next((p for p in ("decode", "full", "prefill") if p in phases), "whole")
    print(f"\n(verdict computed on phase = {phase})")
    caps = [r for r in results if r["mode"] == "cap" and r.get("phase") == phase]
    clks = [r for r in results if r["mode"] == "clock" and r.get("phase") == phase]
    print()
    print("=== Verdict ===")
    if caps:
        engaged = [r for r in caps if r.get("pct_swpowercap", 0) > 5.0]
        pspread = max(r["power_median_w"] for r in caps) - \
                  min(r["power_median_w"] for r in caps)
        if not engaged:
            print(f"  The power cap never engaged (SwPowerCap < 5% at every setting).")
            print(f"  Power spread across caps is {pspread:.1f} W, i.e. incidental variation.")
            print("  => On this card, under this workload, power capping is inert.")
        else:
            lo = min(r["setting"] for r in engaged)
            hi = max(r["setting"] for r in engaged)
            print(f"  The power cap did engage in the {lo}-{hi} W range.")
            print(f"  Power spread across caps is {pspread:.1f} W.")
            print("  => The cap is active in that range, but note it acts by lowering the")
            print("     SM clock. Compare against the clock sweep below to see which")
            print("     reaches lower J per unit of work.")
    if clks:
        best = min(clks, key=lambda r: r["energy_trapz_j"])
        worst = max(clks, key=lambda r: r["energy_trapz_j"])
        if worst["energy_trapz_j"] > 0:
            save = 100 * (1 - best["energy_trapz_j"] / worst["energy_trapz_j"])
            print(f"  Clock sweep: cheapest at {best['setting']} MHz, "
                  f"{save:.1f}% less energy than the most expensive setting.")


# --------------------------------------------------------------------------

def cmd_monitor(args, h):
    s = Sampler(h, args.interval_ms)
    s.start()
    print(f"sampling ({args.interval_ms} ms), Ctrl-C to stop ...")
    try:
        while True:
            time.sleep(1)
            if s.rows:
                r = s.rows[-1]
                print(f"\r{r['power_w']:6.1f} W  {r['sm_mhz']:5} MHz  "
                      f"{r['temp_c']:3}C  util {r['util_pct']:3}%  "
                      f"{','.join(decode_throttle(r['throttle'])) or '-':<40}",
                      end="", flush=True)
    except KeyboardInterrupt:
        pass
    s.stop()
    print("\n")
    print(json.dumps(s.summary(), indent=2))
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(s.rows[0]))
            w.writeheader()
            w.writerows(s.rows)
        print(f"wrote {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["probe", "sweep", "monitor"])
    p.add_argument("-i", "--gpu", type=int, default=0)
    p.add_argument("--interval-ms", type=int, default=50)
    p.add_argument("--out", help="CSV output path")
    p.add_argument("--workload", help="custom workload command. If omitted, use the "
                                      "built-in memory-bound synthetic load (needs torch)")
    p.add_argument("--load-s", type=float, default=20.0, help="synthetic load duration")
    p.add_argument("--tensor-gib", type=float, default=1.0, help="synthetic load tensor size")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--caps", help="comma-separated cap list (W); default is 5 evenly "
                                  "spaced points across the hardware range")
    p.add_argument("--clocks", help="comma-separated SM clock list (MHz)")
    p.add_argument("--test-clamp", action="store_true",
                   help="run the clock clamp test in probe mode")
    p.add_argument("--clamp-load-s", type=float, default=8.0)
    p.add_argument("--no-preflight", action="store_true",
                   help="skip the stock-settings trial run that validates the "
                        "workload before any knob is touched")
    p.add_argument("--preflight-only", action="store_true",
                   help="run the workload exactly once, stream its output to the "
                        "terminal, then exit. Touches no knob and captures nothing; "
                        "use it to debug a failing workload command")
    args = p.parse_args()

    if os.geteuid() == 0 and os.environ.get("SUDO_USER"):
        print("[warn] running under sudo. This tool is designed to run as your "
              "normal user with a NOPASSWD sudoers rule for nvidia-smi; running "
              "the whole thing as root means the workload loses your environment. "
              "See the module docstring.", file=sys.stderr)

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    try:
        {"probe": cmd_probe, "sweep": cmd_sweep, "monitor": cmd_monitor}[args.mode](args, h)
    except PrivilegeError as e:
        print(f"[abort] {e}", file=sys.stderr)
        sys.exit(2)
    finally:
        for k in _ACTIVE_KNOBS:
            k.reset_all()
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
