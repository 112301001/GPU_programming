#!/usr/bin/env python3
"""Roofline plot from the Nsight Compute counters collected by scripts/profile.sh.

    python3 scripts/roofline.py [results/profile/roofline.csv] --peak-gflops 13450 --peak-gbs 616

Defaults are the RTX 2080 Ti (FP32 13.4 TFLOP/s, 616 GB/s), the course demo GPU.
Use your GPU's datasheet numbers (e.g. A30: FP32 10.3 TFLOP/s, 933 GB/s).

FLOPs = FADD + FMUL + 2*FFMA (+ the FP64 equivalents) counted per thread.
Special-function instructions (exp, log, sqrt, sin/cos on the XU pipe) are not
FLOPs here, so the GFLOP/s shown understates the real work done.
"""
import argparse
import csv
import math
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default="results/profile/roofline.csv")
ap.add_argument("--peak-gflops", type=float, default=13450.0)
ap.add_argument("--peak-gbs", type=float, default=616.0)
ap.add_argument("--out", default="results/plots/roofline.png")
args = ap.parse_args()

per_kernel = defaultdict(dict)
with open(args.csv) as f:
    for row in csv.reader(f):
        if len(row) < 3 or row[0] == "ID":
            continue
        # ncu --csv columns: ID, Process ID, Process Name (mc_pricer), ..., Kernel Name, ...,
        # Metric Name, Metric Unit, Metric Value
        kernel = next((c for c in row if c.startswith("mc_") and "kernel" in c), None)
        if kernel is None:
            continue
        name = kernel.split("(")[0]
        metric, value = row[-3], row[-1]
        try:
            per_kernel[name][metric] = float(value.replace(",", ""))
        except ValueError:
            pass

points = []
print(f"{'kernel':22s} {'time ms':>9s} {'GFLOP/s':>9s} {'DRAM GB/s':>10s} {'AI FLOP/B':>10s} {'regs':>5s}")
for name, m in sorted(per_kernel.items()):
    g = lambda k: m.get(k, 0.0)
    flops = (g("sm__sass_thread_inst_executed_op_fadd_pred_on.sum") + g("sm__sass_thread_inst_executed_op_fmul_pred_on.sum")
             + 2 * g("sm__sass_thread_inst_executed_op_ffma_pred_on.sum")
             + g("sm__sass_thread_inst_executed_op_dadd_pred_on.sum") + g("sm__sass_thread_inst_executed_op_dmul_pred_on.sum")
             + 2 * g("sm__sass_thread_inst_executed_op_dfma_pred_on.sum"))
    t_ns = g("gpu__time_duration.sum")
    nbytes = g("dram__bytes.sum")
    if t_ns <= 0:
        continue
    gflops = flops / t_ns
    gbs = nbytes / t_ns
    ai = flops / nbytes if nbytes > 0 else math.inf
    points.append((name, ai, gflops))
    print(f"{name:22s} {t_ns / 1e6:9.3f} {gflops:9.1f} {gbs:10.2f} {ai:10.1f} {g('launch__registers_per_thread'):5.0f}")

if not points:
    raise SystemExit(f"no kernel data in {args.csv}: Nsight Compute did not collect any counters "
                     "(see the messages printed by scripts/profile.sh)")

ridge = args.peak_gflops / args.peak_gbs
print(f"ridge point = {ridge:.1f} FLOP/byte (kernels to the right are compute-bound)")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib not installed; table above only")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
xs = [10 ** (k / 10) for k in range(-20, 61)]
ax.loglog(xs, [min(args.peak_gflops, x * args.peak_gbs) for x in xs], "k-", lw=2, label="roofline")
for name, ai, gf in points:
    x = ai if math.isfinite(ai) else 1e6
    ax.loglog([x], [gf], "o", ms=8, label=f"{name} (AI={'inf' if not math.isfinite(ai) else f'{ai:.0f}'})")
ax.axvline(ridge, color="gray", ls=":", lw=1)
ax.set_xlabel("arithmetic intensity (FLOP / DRAM byte)")
ax.set_ylabel("performance (GFLOP/s)")
ax.set_title("Roofline (FP32 peak %.0f GFLOP/s, %.0f GB/s)" % (args.peak_gflops, args.peak_gbs))
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print("wrote", args.out)
