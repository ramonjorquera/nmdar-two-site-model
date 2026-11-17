# -*- coding: utf-8 -*-
"""
nmdar_sensitivity_oat.py — Local (one-at-a-time) sensitivity of the NMDAR model.

Companion code for Jorquera & Stehberg (NMDAR convergent-pharmacology modeling
paper), Section 3.6.

Two analyses:
  1. Tornado: each of the 19 parameters is perturbed +/- 50 % about its canonical
     value (clipped to its physical domain), one at a time, and the convergence
     fold O1 = P_conv / P_healthy is recomputed. Reports the low/high value and
     the swing for every parameter, sorted by swing. The baseline fold is 2.64.
  2. Crossover vs Ki: the homocysteine/D-serine crossover threshold (the D-serine
     concentration at which the sign of the homocysteine effect inverts) is mapped
     against the homocysteine glycine-site antagonism constant Ki across its range.

Outputs (wide-format CSV, same folder):
    sens_tornado.csv          parameter, O1_low, O1_high, swing  (sorted)
    sens_crossover_vs_Ki.csv  Ki (uM), crossover D-serine (uM)

Requirements: Python 3, numpy. Reuses nmdar_model and nmdar_sensitivity_lhs.
Usage:        python nmdar_sensitivity_oat.py
"""

import csv
import numpy as np
from nmdar_model import Popen
from nmdar_sensitivity_lhs import SPECS, NAMES, model_params, conditions, crossover_dser

# canonical values and bounds from the parameter table
CANON = {name: canon for (name, canon, lo, hi, dist, role) in SPECS}
FRACTION_PARAMS = {'phi_infl', 'eps_Hcy', 'eps_QUIN', 'w_2A'}   # clipped to [0,1]


def O1_fold(sample):
    """Convergence fold P_conv / P_healthy for a full sample dict."""
    pr = model_params(sample)
    c = conditions(sample)
    Ph = Popen(params=pr, **c['Healthy'])
    Pc = Popen(params=pr, **c['Convergence'])
    return Pc / Ph


def tornado():
    base = O1_fold(CANON)
    rows = []
    for name in NAMES:
        vals = []
        for factor in (0.5, 1.5):
            s = dict(CANON)
            v = CANON[name] * factor
            if name in FRACTION_PARAMS:
                v = min(max(v, 0.0), 1.0)
            s[name] = v
            vals.append(O1_fold(s))
        lo, hi = min(vals), max(vals)
        rows.append((name, lo, hi, hi - lo))
    rows.sort(key=lambda r: r[3], reverse=True)

    print(f"Tornado: convergence fold O1 under +/-50 % OAT (baseline = {base:.2f})\n")
    print(f"  {'parameter':12s} {'low':>7s} {'high':>7s} {'swing':>7s}")
    for name, lo, hi, sw in rows:
        print(f"  {name:12s} {lo:7.2f} {hi:7.2f} {sw:7.2f}")
    top = rows[0]
    print(f"\nlargest excursion: {top[0]} ({top[1]:.2f}-{top[2]:.2f}); "
          f"no single parameter abolishes the ~2x amplification.")

    with open('sens_tornado.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['parameter', 'O1_low', 'O1_high', 'swing', 'baseline'])
        for name, lo, hi, sw in rows:
            w.writerow([name, f"{lo:.4f}", f"{hi:.4f}", f"{sw:.4f}", f"{base:.4f}"])
    return base, rows


def crossover_vs_Ki(Ki_values=(5, 7.5, 10, 15, 20, 30, 40, 50)):
    pr0 = model_params(CANON)
    print("\nCrossover D-serine vs Ki:")
    print(f"  {'Ki (uM)':>8s} {'crossover Dser (uM)':>20s}")
    out = []
    for Ki in Ki_values:
        pr = dict(pr0)
        pr['Ki'] = float(Ki)
        d = crossover_dser(pr)
        out.append((Ki, d))
        print(f"  {Ki:8.1f} {d:20.3f}")
    with open('sens_crossover_vs_Ki.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['Ki_uM', 'crossover_Dserine_uM'])
        for Ki, d in out:
            w.writerow([f"{Ki:.2f}", f"{d:.4f}"])
    canon_d = crossover_dser(pr0)
    print(f"\n  canonical (Ki = {CANON['Ki']:.0f}) crossover = {canon_d:.3f} uM")
    return out


if __name__ == '__main__':
    tornado()
    crossover_vs_Ki()
    print("\nwrote sens_tornado.csv, sens_crossover_vs_Ki.csv")
