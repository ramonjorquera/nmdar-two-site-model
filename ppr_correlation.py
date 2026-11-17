# -*- coding: utf-8 -*-
"""
ppr_correlation.py — Model-vs-literature paired-pulse ratio (PPR) correlation
for the Supplementary "Regional vulnerability" section (Jorquera & Stehberg,
NMDAR convergent-pharmacology modeling paper; Table S-reg / Fig S-PPR).

Six forebrain circuits: model PPR (Tsodyks-Markram release model with regional
U, tau_rec, tau_fac) vs the experimental literature value (point, or midpoint of
a reported range with low/high for horizontal error bars). Reports Pearson r, R2,
the regression of literature on model, and the deviation from the identity line.
Closed-form statistics (no scipy / no LAPACK).

Output: PPR_correlation.csv   (region, PPR_model, PPR_lit, lit_low, lit_high, note)
Usage:  python ppr_correlation.py
"""

import numpy as np

# region, PPR_model, lit_point, lit_low, lit_high, note
ROWS = [
    ("Hippocampus CA1",      1.51, 1.75, 1.50, 2.00, "range 1.5-2.0 (midpoint)"),
    ("mPFC L2/3",            1.48, 1.30, 1.00, 1.50, "reported >1 (weakly constrained)"),
    ("Amygdala LA cortical", 1.23, 1.22, 1.22, 1.22, "point (Guo2017 slice)"),
    ("Amygdala LA thalamic", 1.10, 1.10, 1.10, 1.10, "point (Guo2017 slice)"),
    ("Dorsal striatum",      0.82, 0.90, 0.80, 1.00, "range 0.8-1.0 (midpoint)"),
    ("Visual cortex V1",     0.62, 0.63, 0.63, 0.63, "point"),
]

m = np.array([r[1] for r in ROWS])   # model PPR
l = np.array([r[2] for r in ROWS])   # literature PPR

mm, lm = m.mean(), l.mean()
r = ((m - mm) * (l - lm)).sum() / np.sqrt(((m - mm) ** 2).sum() * ((l - lm) ** 2).sum())
b = ((m - mm) * (l - lm)).sum() / ((m - mm) ** 2).sum()   # slope (lit on model)
a = lm - b * mm                                           # intercept
mae = np.abs(m - l).mean()
rmse_id = np.sqrt(((m - l) ** 2).mean())                  # vs identity (y = x)

with open("PPR_correlation.csv", "w", encoding="utf-8") as f:
    f.write("region,PPR_model,PPR_lit,lit_low,lit_high,lit_note\n")
    for rr in ROWS:
        f.write("%s,%.3f,%.3f,%.3f,%.3f,%s\n" % rr)

print("wrote PPR_correlation.csv")
print("Pearson r = %.3f | R2 = %.3f" % (r, r * r))
print("regression: lit = %.3f + %.3f*model" % (a, b))
print("mean |model-lit| = %.3f ; RMSE vs identity = %.3f" % (mae, rmse_id))
print("range covered: model %.2f-%.2f, lit %.2f-%.2f" % (m.min(), m.max(), l.min(), l.max()))
