# -*- coding: utf-8 -*-
"""
nmdar_model.py — Two-site occupancy x efficacy model of the NMDA receptor.

Companion code for:
    Jorquera, R. A. & Stehberg, J. "Convergent pharmacology at the two agonist
    sites of the NMDA receptor: homocysteine, quinolinic acid and co-agonist
    tone as a permissive amplifier of excitotoxic signaling."

This module is the canonical implementation of the model described in Section 2
of the manuscript. It computes the steady-state ambient open probability (Popen)
of a forebrain GluN2A/2B receptor pool as a function of the metabolic-inflammatory
milieu, and reproduces the central results of the paper (Table 1 and the
supra-additivity of the convergent condition).

The model math is parameterized: Popen() accepts an optional `params` dict that
overrides any entry of DEFAULT_PARAMS (the canonical values). This single source
of the math is reused by the sensitivity scripts (nmdar_sensitivity_lhs.py,
nmdar_sensitivity_oat.py).

Model summary (strict-coincidence / product regime):
  - Glutamate-site ligands (glutamate, homocysteic acid HCA, quinolinic acid QUIN,
    homocysteine Hcy) contribute a normalized drive f_i = [L_i]/Kd_i and an
    intrinsic efficacy eps_i. A drive-weighted mean efficacy ebar scales activation.
  - Hcy is a weak, self-gated partial agonist (eps = 0.3): a glutamate-site agonist
    that is simultaneously a glycine-site partial antagonist (constant Ki).
  - The glutamate site is cooperative (Hill n_g) and negatively coupled (kappa) to
    the co-agonist (GluN1) site; both activations are solved self-consistently by
    fixed-point iteration.
  - A redox fraction phi partitions the total homocysteine pool HcyT between the
    weak parent (Hcy) and the potent acid (HCA): HCA = 1 + phi*HcyT,
    Hcy = 1 + (1-phi)*HcyT. phi tracks inflammation/ROS; HcyT tracks folate status.
  - Popen sums the two subunit sub-populations weighted by occupancy (w) and
    single-channel conductance (G), through a saturating map sat().

Parameter provenance: NMDAR model appendix v8 (tables 0, 2, 3); pharmacology
anchored to the companion review and to independent recombinant-receptor data
(see manuscript Section 5 and Table S-sens).

Requirements: Python 3, numpy.
Usage:        python nmdar_model.py          # prints Table 1 + supra-additivity
License:      MIT (see LICENSE)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Canonical parameters (appendix v8, tables 0 & 3). These are the DEFAULTS;
# any subset can be overridden per call via the `params` argument of Popen().
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    'Kd_Glu': 3.0, 'Kd_HCA': 20.0, 'Kd_QUIN': 30.0,   # glutamate-site Kd (uM)
    'Kd_Hcy_2A': 10.0, 'Kd_Hcy_2B': 60.0,             # subunit-specific Hcy Kd (uM)
    'Kd_Dser': 0.5, 'Kd_Gly': 0.3,                    # co-agonist Kd (uM)
    'eps_Glu': 1.0, 'eps_HCA': 1.0,                   # full-agonist efficacies
    'eps_QUIN': 1.0, 'eps_Hcy': 0.3,                  # QUIN weak, Hcy partial (0.3)
    'n_g': 1.3,                                        # glutamate-site Hill coeff.
    'kappa': 1.0,                                      # glu <-> co-agonist coupling
    'Ksat': 0.30,                                      # saturation constant
    'Ki': 15.0,                                        # Hcy glycine-site antag. (uM)
    'w_2A': 0.6,                                        # subunit occupancy weight 2A
    'G_2A': 0.805, 'G_2B': 0.016,                      # single-channel conductance
    'Gly_ambient': 0.1,                                # ambient glycine (uM)
}


def _merge(params):
    """Return a full parameter dict: DEFAULT_PARAMS updated by `params` (if any)."""
    if not params:
        return DEFAULT_PARAMS
    pr = dict(DEFAULT_PARAMS)
    pr.update(params)
    return pr


def _subtype_terms(p, Glu, HCA, QUIN, Hcy, Dser, Gly, pr):
    """Self-consistent (ogeff, theta_c) for one subunit p in {'2A','2B'}.

    ogeff = ebar * theta_g  (efficacy-weighted glutamate-site activation);
    theta_c is the co-agonist-site occupancy. Solved by fixed-point iteration.
    """
    kH = pr['Kd_Hcy_2A'] if p == '2A' else pr['Kd_Hcy_2B']
    # glutamate-site normalized binding terms f_i = [L_i]/Kd_i
    f_Glu = Glu / pr['Kd_Glu']
    f_HCA = HCA / pr['Kd_HCA']
    f_QUIN = QUIN / pr['Kd_QUIN']
    f_Hcy = Hcy / kH
    S = f_Glu + f_HCA + f_QUIN + f_Hcy
    ebar = (pr['eps_Glu'] * f_Glu + pr['eps_HCA'] * f_HCA +
            pr['eps_QUIN'] * f_QUIN + pr['eps_Hcy'] * f_Hcy) / S
    A = Dser / pr['Kd_Dser'] + Gly / pr['Kd_Gly']     # co-agonist drive
    kappa, n_g, Ki = pr['kappa'], pr['n_g'], pr['Ki']
    tg, tc = 0.0, 0.0
    for _ in range(200):
        Sc = S / (1.0 + kappa * tc)
        Ac = A / (1.0 + kappa * tg)
        tg_new = Sc ** n_g / (1.0 + Sc ** n_g)
        tc_new = Ac / (1.0 + Ac + Hcy / Ki)
        if abs(tg_new - tg) < 1e-12 and abs(tc_new - tc) < 1e-12:
            tg, tc = tg_new, tc_new
            break
        tg, tc = tg_new, tc_new
    return ebar * tg, tc


def Popen(phi, HcyT, Dser, QUIN, Glu, Gly=None, params=None):
    """Steady-state ambient open probability of the GluN2A/2B pool.

    Parameters
    ----------
    phi    : redox/oxidation fraction partitioning HcyT into HCA (0..1)
    HcyT   : total homocysteine pool (uM, arbitrary occupancy units)
    Dser   : D-serine (uM)
    QUIN   : quinolinic acid (uM)
    Glu    : glutamate (arbitrary occupancy units)
    Gly    : glycine (uM); defaults to ambient value in params
    params : optional dict overriding any entry of DEFAULT_PARAMS
    """
    pr = _merge(params)
    if Gly is None:
        Gly = pr['Gly_ambient']
    HCA = 1.0 + phi * HcyT
    Hcy = 1.0 + (1.0 - phi) * HcyT
    w = {'2A': pr['w_2A'], '2B': 1.0 - pr['w_2A']}
    Gc = {'2A': pr['G_2A'], '2B': pr['G_2B']}
    Ksat = pr['Ksat']
    P = 0.0
    for p in ('2A', '2B'):
        og, tc = _subtype_terms(p, Glu, HCA, QUIN, Hcy, Dser, Gly, pr)
        x = og * tc
        P += w[p] * Gc[p] * (x / (Ksat + x))
    return P


def subunit_popen(subtype, phi, HcyT, Dser, QUIN, Glu, Gly=None, params=None):
    """Per-subunit open probability for one subunit pool ('2A' or '2B').

    Returns (P_intrinsic, contribution) where P_intrinsic = sat(ogeff*theta_c)
    is the subunit's own open probability and contribution = w*G*P_intrinsic is
    its weighted share of the total Popen. Used for Fig 2C (the GluN2A-vs-2B
    homocysteine-sensitivity panel).
    """
    pr = _merge(params)
    if Gly is None:
        Gly = pr['Gly_ambient']
    HCA = 1.0 + phi * HcyT
    Hcy = 1.0 + (1.0 - phi) * HcyT
    og, tc = _subtype_terms(subtype, Glu, HCA, QUIN, Hcy, Dser, Gly, pr)
    x = og * tc
    P_intr = x / (pr['Ksat'] + x)
    w = pr['w_2A'] if subtype == '2A' else 1.0 - pr['w_2A']
    Gc = pr['G_2A'] if subtype == '2A' else pr['G_2B']
    return P_intr, w * Gc * P_intr


# ---------------------------------------------------------------------------
# The four milieu conditions (appendix v8, table 2 / manuscript Table 1)
# ---------------------------------------------------------------------------
CONDITIONS = {
    'Healthy':         dict(phi=0.1, HcyT=2,  Dser=0.15, QUIN=0.2, Glu=1.0),
    'Inflammation':    dict(phi=0.7, HcyT=2,  Dser=1.2,  QUIN=4,   Glu=1.5),
    'MTHF_deficiency': dict(phi=0.1, HcyT=25, Dser=0.15, QUIN=0.2, Glu=1.0),
    'Convergence':     dict(phi=0.7, HcyT=25, Dser=1.2,  QUIN=4,   Glu=1.5),
}

# Published reference values (manuscript Table 1)
REF = {'Healthy': 0.0845, 'Inflammation': 0.1841,
       'MTHF_deficiency': 0.0579, 'Convergence': 0.2227}


def condition_popen(params=None):
    """Return {condition: Popen} for the four canonical milieu conditions."""
    return {name: Popen(params=params, **c) for name, c in CONDITIONS.items()}


def table1(save_csv=None):
    """Print Table 1 (Popen, fold vs healthy) and the supra-additivity summary."""
    P = condition_popen()
    h = P['Healthy']
    add = h + (P['Inflammation'] - h) + (P['MTHF_deficiency'] - h)
    excess = P['Convergence'] - add

    print(f"{'condition':18s} {'Popen':>8s} {'fold':>6s} {'ref':>8s} {'dP':>8s}")
    for name in CONDITIONS:
        print(f"{name:18s} {P[name]:8.4f} {P[name]/h:6.2f} "
              f"{REF[name]:8.4f} {P[name]-REF[name]:+8.4f}")
    print()
    print(f"additive expectation  = {add:.4f} (fold {add/h:.2f})")
    print(f"convergence observed  = {P['Convergence']:.4f} "
          f"(fold {P['Convergence']/h:.2f})")
    print(f"supra-additive excess = {excess:+.4f}  "
          f"({100*excess/add:+.1f}% over additive)")

    if save_csv:
        import csv
        with open(save_csv, 'w', newline='', encoding='utf-8') as fh:
            wtr = csv.writer(fh)
            wtr.writerow(['condition', 'phi', 'HcyT', 'D_serine', 'QUIN',
                          'Glu', 'Popen', 'fold_vs_healthy'])
            for name, c in CONDITIONS.items():
                wtr.writerow([name, c['phi'], c['HcyT'], c['Dser'],
                              c['QUIN'], c['Glu'], f"{P[name]:.6f}",
                              f"{P[name]/h:.4f}"])
            wtr.writerow([])
            wtr.writerow(['additive_prediction', '', '', '', '', '',
                          f"{add:.6f}", f"{add/h:.4f}"])
            wtr.writerow(['supra_additive_excess', '', '', '', '', '',
                          f"{excess:.6f}", f"{100*excess/add:+.2f}%"])
        print(f"\nwrote {save_csv}")
    return P


if __name__ == '__main__':
    table1(save_csv='table1_popen.csv')
