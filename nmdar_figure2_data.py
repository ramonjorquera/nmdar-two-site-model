# -*- coding: utf-8 -*-
"""
nmdar_figure2_data.py — Regenerate the data behind Figure 2 of the NMDAR
convergent-pharmacology modeling paper (Jorquera & Stehberg).

All panels use the canonical model (nmdar_model.py) in a low-oxidation reference
milieu (phi = 0.1, QUIN = 0.2, Glu = 1.0) unless a panel varies one of these,
because that is the regime in which the homocysteine pool stays mostly as the
inhibitory parent and the sign reversal lives.

Panels / outputs (same filenames the Origin figure imports):
  A  Lipton crossover curve        fig2B_crossover_curve.csv
     (+ Popen-vs-Hcy family)       fig2B_popen_vs_hcy_family.csv
  B  Hcy x D-serine gate heatmap   fig2C_popen_matrix.dat, fig2C_pctchange_matrix.dat,
                                    fig2C_axes.csv
  C  GluN2A vs GluN2B Hcy sens.     subunit_hcy_sensitivity.csv
  D  Hcy x phi (HCA) heatmap        fig2phi_popen_matrix.dat, fig2phi_pctchange_matrix.dat,
                                    fig2phi_axes.csv
  (supp) Popen vs Hcy at several phi   phi_hcy_panel.csv

Requirements: Python 3, numpy. Usage: python nmdar_figure2_data.py
"""

import numpy as np
from nmdar_model import Popen, subunit_popen

# low-oxidation reference milieu
PHI0, QUIN0, GLU0 = 0.1, 0.2, 1.0


def _zero_crossing(x, y):
    """First x where y crosses from < 0 to >= 0 (NaN if none)."""
    for i in range(len(x) - 1):
        if y[i] < 0 <= y[i + 1]:
            return x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    return np.nan


# ---------------------------------------------------------------------------
# Panel A — Lipton crossover: %change of Popen (HcyT 1 -> 30) vs D-serine
# ---------------------------------------------------------------------------
def panel_A():
    def P(HcyT, Dser):
        return Popen(PHI0, HcyT, Dser, QUIN0, GLU0)
    Dser = np.linspace(0.05, 3.0, 120)
    pct = np.array([100.0 * (P(30.0, d) - P(1.0, d)) / P(1.0, d) for d in Dser])
    with open('fig2B_crossover_curve.csv', 'w', encoding='utf-8') as f:
        f.write('Dserine_uM,pct_change_Popen_Hcy1to30\n')
        for d, y in zip(Dser, pct):
            f.write(f'{d:.4f},{y:.4f}\n')

    HcyT = np.linspace(1.0, 30.0, 60)
    Dlevels = [0.15, 0.5, 1.2, 2.0]
    with open('fig2B_popen_vs_hcy_family.csv', 'w', encoding='utf-8') as f:
        f.write('HcyT_uM,' + ','.join(f'Popen_Dser{d:g}' for d in Dlevels) + '\n')
        for h in HcyT:
            f.write(f'{h:.3f},' + ','.join(f'{P(h, d):.5f}' for d in Dlevels) + '\n')
    print(f"A  crossover (pct=0) at D-serine = {_zero_crossing(Dser, pct):.3f} uM")


# ---------------------------------------------------------------------------
# Panel B — Hcy x D-serine gate heatmap (%change vs Hcy=1 per D-serine column)
# ---------------------------------------------------------------------------
def panel_B():
    def P(HcyT, Dser):
        return Popen(PHI0, HcyT, Dser, QUIN0, GLU0)
    HcyT_g = np.linspace(1.0, 30.0, 20)      # rows (Y)
    Dser_g = np.linspace(0.10, 2.50, 20)     # cols (X)
    popen = np.array([[P(h, d) for d in Dser_g] for h in HcyT_g])
    base = np.array([P(1.0, d) for d in Dser_g])
    pct = 100.0 * (popen - base[None, :]) / base[None, :]
    np.savetxt('fig2C_popen_matrix.dat', popen, fmt='%.5f', delimiter='\t')
    np.savetxt('fig2C_pctchange_matrix.dat', pct, fmt='%.3f', delimiter='\t')
    with open('fig2C_axes.csv', 'w', encoding='utf-8') as f:
        f.write('row_index,HcyT_uM(Y)\n')
        for i, h in enumerate(HcyT_g, 1):
            f.write(f'{i},{h:.3f}\n')
        f.write('col_index,Dserine_uM(X)\n')
        for j, d in enumerate(Dser_g, 1):
            f.write(f'{j},{d:.3f}\n')
    print(f"B  Hcy x D-serine: pct-change range {pct.min():.1f}% to {pct.max():.1f}%")


# ---------------------------------------------------------------------------
# Panel C — GluN2A vs GluN2B homocysteine sensitivity (Kd_Hcy 10 vs 60)
# ---------------------------------------------------------------------------
def panel_C():
    Dser = 0.15  # low-D-serine milieu (MTHF-deficiency axis)
    HcyT = np.arange(1, 31, 1, dtype=float)
    P1 = {p: subunit_popen(p, PHI0, 1.0, Dser, QUIN0, GLU0)[0] for p in ('2A', '2B')}
    with open('subunit_hcy_sensitivity.csv', 'w', encoding='utf-8') as f:
        f.write('HcyT_uM,P_GluN2A,P_GluN2B,Pnorm_GluN2A,Pnorm_GluN2B,'
                'contrib_GluN2A,contrib_GluN2B\n')
        last = None
        for h in HcyT:
            P2A, c2A = subunit_popen('2A', PHI0, h, Dser, QUIN0, GLU0)
            P2B, c2B = subunit_popen('2B', PHI0, h, Dser, QUIN0, GLU0)
            f.write(f'{h:g},{P2A:.5f},{P2B:.5f},{P2A/P1["2A"]:.4f},'
                    f'{P2B/P1["2B"]:.4f},{c2A:.5f},{c2B:.5f}\n')
            last = (P2A, P2B)
    d2A = 100 * (last[0] - P1['2A']) / P1['2A']
    d2B = 100 * (last[1] - P1['2B']) / P1['2B']
    print(f"C  GluN2A {d2A:+.0f}% vs GluN2B {d2B:+.0f}% (HcyT 1->30)")


# ---------------------------------------------------------------------------
# Panel D — Hcy x phi (oxidation to HCA) heatmap
# ---------------------------------------------------------------------------
def panel_D():
    Dser = 0.15
    HcyT_g = np.linspace(1.0, 30.0, 20)      # rows (Y)
    phi_g = np.linspace(0.1, 0.7, 20)        # cols (X)
    popen = np.array([[Popen(p, h, Dser, QUIN0, GLU0) for p in phi_g] for h in HcyT_g])
    base = np.array([Popen(p, 1.0, Dser, QUIN0, GLU0) for p in phi_g])
    pct = 100.0 * (popen - base[None, :]) / base[None, :]
    np.savetxt('fig2phi_popen_matrix.dat', popen, fmt='%.5f', delimiter='\t')
    np.savetxt('fig2phi_pctchange_matrix.dat', pct, fmt='%.3f', delimiter='\t')
    with open('fig2phi_axes.csv', 'w', encoding='utf-8') as f:
        f.write('row_index,HcyT_uM(Y)\n')
        for i, h in enumerate(HcyT_g, 1):
            f.write(f'{i},{h:.3f}\n')
        f.write('col_index,phi(X)\n')
        for j, p in enumerate(phi_g, 1):
            f.write(f'{j},{p:.3f}\n')
    flip = _zero_crossing(phi_g, pct[-1, :])   # at HcyT=30
    print(f"D  Hcy x phi: pct range {pct.min():.1f}% to {pct.max():.1f}%; "
          f"sign flip (HcyT=30) near phi = {flip:.3f}")


# ---------------------------------------------------------------------------
# Supplementary — Popen vs Hcy at several phi
# ---------------------------------------------------------------------------
def phi_panel():
    Dser = 0.15
    phis = [0.1, 0.3, 0.5, 0.7]
    HcyT = np.arange(1, 31, 1, dtype=float)
    cols = {p: [Popen(p, h, Dser, QUIN0, GLU0) for h in HcyT] for p in phis}
    with open('phi_hcy_panel.csv', 'w', encoding='utf-8') as f:
        f.write('HcyT_uM,' + ','.join(f'Popen_phi{p:.1f}' for p in phis) + '\n')
        for i, h in enumerate(HcyT):
            f.write(f'{h:g},' + ','.join(f'{cols[p][i]:.5f}' for p in phis) + '\n')
    print("supp  phi_hcy_panel.csv written")


if __name__ == '__main__':
    panel_A()
    panel_B()
    panel_C()
    panel_D()
    phi_panel()
    print("\nFigure 2 data regenerated.")
