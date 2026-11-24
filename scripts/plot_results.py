#!/usr/bin/env python3
"""Plot the CSV files written by scripts/benchmark.sh into results/plots/*.png.

Needs matplotlib (pip install matplotlib). Missing CSV files are skipped.
"""
import csv
import math
import os
import sys
from collections import defaultdict

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = os.path.join(RES, "plots")
os.makedirs(OUT, exist_ok=True)


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        print(f"skip {name} (not found)")
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("paths", "steps", "block", "blocks_per_sm", "antithetic"):
            r[k] = int(r[k])
        for k in ("price", "std_error", "time_ms", "kernel_ms", "mpaths_per_s"):
            r[k] = float(r[k])
        r["abs_error"] = float(r["abs_error"]) if r["abs_error"] else math.nan
    return rows


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


# 1. convergence -----------------------------------------------------------
rows = load("convergence.csv")
if rows:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for anti, label in ((0, "plain"), (1, "antithetic")):
        rs = sorted((r for r in rows if r["antithetic"] == anti), key=lambda r: r["paths"])
        n = [r["paths"] for r in rs]
        ax.loglog(n, [r["std_error"] for r in rs], "o-", label=f"std error ({label})")
        ax.loglog(n, [r["abs_error"] for r in rs], "x--", alpha=0.6, label=f"|price - BS| ({label})")
    n0 = min(r["paths"] for r in rows)
    se0 = max(r["std_error"] for r in rows if r["paths"] == n0)
    ns = sorted({r["paths"] for r in rows})
    ax.loglog(ns, [se0 * math.sqrt(n0 / x) for x in ns], "k:", label=r"$\propto N^{-1/2}$")
    ax.set_xlabel("simulated paths N")
    ax.set_ylabel("error")
    ax.set_title("Convergence (European call, gpu-shuffle)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    save(fig, "convergence.png")

# 2. scaling ---------------------------------------------------------------
rows = load("scaling.csv")
if rows:
    by = defaultdict(list)
    for r in rows:
        by[r["engine"]].append(r)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for eng, rs in by.items():
        rs.sort(key=lambda r: r["paths"])
        a1.loglog([r["paths"] for r in rs], [r["time_ms"] for r in rs], "o-", label=eng)
        a2.semilogx([r["paths"] for r in rs], [r["mpaths_per_s"] for r in rs], "o-", label=eng)
    a1.set_xlabel("paths N")
    a1.set_ylabel("time (ms, end to end)")
    a1.set_title("Run time vs N (252 steps)")
    a2.set_xlabel("paths N")
    a2.set_ylabel("throughput (M paths/s)")
    a2.set_yscale("log")
    a2.set_title("Throughput vs N")
    for a in (a1, a2):
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    save(fig, "scaling.png")

# 3. ablation --------------------------------------------------------------
rows = load("ablation.csv")
if rows:
    gpu = [r for r in rows if r["engine"].startswith("gpu")]
    gpu.sort(key=lambda r: r["mpaths_per_s"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = [f'{r["engine"]}\n(N={r["paths"]:.0e})' for r in gpu]
    ax.barh(names, [r["mpaths_per_s"] for r in gpu], color="tab:blue")
    for i, r in enumerate(gpu):
        ax.text(r["mpaths_per_s"], i, f' {r["mpaths_per_s"]:.0f}', va="center", fontsize=8)
    ax.set_xlabel("throughput (M paths/s, end to end)")
    ax.set_title("Ablation: one design decision changed per engine")
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "ablation.png")

# 4. antithetic ------------------------------------------------------------
rows = load("antithetic.csv")
if rows:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels, ratios = [], []
    for style in ("european", "asian"):
        for eng in ("cpu", "gpu-shuffle"):
            se = {r["antithetic"]: r["std_error"] for r in rows if r["style"] == style and r["engine"] == eng}
            if 0 in se and 1 in se:
                labels.append(f"{style}\n{eng}")
                ratios.append((se[0] / se[1]) ** 2)
    ax.bar(labels, ratios, color="tab:green")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_ylabel("variance reduction factor (SE_plain / SE_anti)^2")
    ax.set_title("Antithetic variates at equal path count")
    save(fig, "antithetic.png")

# 5. launch configuration --------------------------------------------------
rows = load("launch.csv")
if rows:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for bs in sorted({r["block"] for r in rows}):
        rs = sorted((r for r in rows if r["block"] == bs), key=lambda r: r["blocks_per_sm"])
        ax.semilogx([r["blocks_per_sm"] for r in rs], [r["mpaths_per_s"] for r in rs], "o-", base=2,
                    label=f"{bs} threads/block")
    ax.set_xlabel("blocks per SM (grid = blocks/SM x #SMs)")
    ax.set_ylabel("throughput (M paths/s)")
    ax.set_title("Launch configuration (gpu-shuffle)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "launch.png")

# 6. styles ----------------------------------------------------------------
rows = load("styles.csv")
if rows:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    combos = sorted({(r["style"], r["scheme"]) for r in rows})
    x = range(len(combos))
    for k, eng in enumerate(("cpu", "gpu-shuffle")):
        vals = []
        for st, sc in combos:
            m = [r["mpaths_per_s"] for r in rows if r["style"] == st and r["scheme"] == sc and r["engine"] == eng]
            vals.append(m[0] if m else 0)
        ax.bar([i + 0.4 * k for i in x], vals, width=0.4, label=eng)
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels([f"{st}\n{sc}" for st, sc in combos], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("throughput (M paths/s)")
    ax.set_title("Option styles and discretisation schemes (252 steps)")
    ax.legend()
    save(fig, "styles.png")

# 7. steps -----------------------------------------------------------------
rows = load("steps.csv")
if rows:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for eng in sorted({r["engine"] for r in rows}):
        rs = sorted((r for r in rows if r["engine"] == eng), key=lambda r: r["steps"])
        ax.loglog([r["steps"] for r in rs], [r["kernel_ms"] for r in rs], "o-", label=f"{eng} kernel")
        ax.loglog([r["steps"] for r in rs], [r["time_ms"] for r in rs], "x--", label=f"{eng} total")
    ax.set_xlabel("time steps per path M")
    ax.set_ylabel("time (ms)")
    ax.set_title("Cost vs time steps (arithmetic Asian)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "steps.png")
