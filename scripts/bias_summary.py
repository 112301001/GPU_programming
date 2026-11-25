#!/usr/bin/env python3
"""Summarise results/bias.csv written by scripts/bias_check.sh (standard library only).

For each (engine, scheme): over K seeds, z_i = (price_i - reference) / se_i.
  mean z     should be within about +-2/sqrt(K) of 0 for an unbiased engine
  t = mean z * sqrt(K)   |t| > 3 means a real bias
  sd z       should be about 1 (checks that the reported standard error is right)
  mean err   average price - reference (the size of any bias, in price units)
"""
import csv
import math
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results/bias.csv"
groups = defaultdict(list)
with open(path) as f:
    for r in csv.DictReader(f):
        if not r["reference"]:
            continue
        price, ref, se = float(r["price"]), float(r["reference"]), float(r["std_error"])
        groups[(r["scheme"], r["engine"])].append((price - ref, (price - ref) / se))

print(f"{'scheme':6s} {'engine':12s} {'K':>3s} {'mean z':>8s} {'t':>7s} {'sd z':>6s} {'mean err':>10s}  verdict")
for (scheme, engine), v in sorted(groups.items()):
    k = len(v)
    zs = [z for _, z in v]
    errs = [e for e, _ in v]
    mz = sum(zs) / k
    sdz = math.sqrt(sum((z - mz) ** 2 for z in zs) / (k - 1)) if k > 1 else float("nan")
    t = mz * math.sqrt(k)
    verdict = "ok" if abs(t) < 3 else "BIASED"
    print(f"{scheme:6s} {engine:12s} {k:3d} {mz:+8.3f} {t:+7.2f} {sdz:6.2f} {sum(errs) / k:+10.5f}  {verdict}")
