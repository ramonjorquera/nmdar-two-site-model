# -*- coding: utf-8 -*-
"""
nmdar_sensitivity_lhs.py — Global sensitivity analysis of the NMDAR two-site model.

Companion code for Jorquera & Stehberg (NMDAR convergent-pharmacology modeling
paper), Section 3 (Results) / Table S-sens.

Method (manuscript Section 2, "Sensitivity analysis"):
  - 19 parameters varied simultaneously within literature-derived ranges by
    Latin-hypercube sampling (default N = 3000, fixed seed for reproducibility).
  - For every sample the four milieu conditions are recomputed and four summary
    metrics are extracted, one per central claim:
        O1  convergence fold  P_conv / P_healthy          (claim: > 1)
        O2  supra-additive excess (P_conv - P_add)/P_add  (claim: > 0)
        O3  deficiency fold   P_def / P_healthy           (claim: <= 1)
        O4  D-serine crossover: [D-serine] at which the sign of the homocysteine
            effect (Popen at HcyT=30 minus HcyT=1, low-oxidation context) inverts
                                                          (claim: a crossover
            exists within the physiological D-serine range)
  - Reports the fraction of the sampled space in which each claim holds, metric
    medians and 5-95 percentiles, and the partial rank correlation coefficient
    (PRCC) of every parameter on every metric (precision-matrix method on the
    rank-transformed samples; matrix inverse by Gauss-Jordan, no LAPACK).

Outputs (wide-format CSV, same folder):
    sens_lhs_outputs.csv   per-sample inputs (19) + O1..O4
    sens_prcc.csv          19 x 4 PRCC matrix
    sens_summary.csv       claim fractions, medians, 5-95 percentiles per metric

Requirements: Python 3, numpy (no scipy / no LAPACK).
Usage:        python nmdar_sensitivity_lhs.py [N] [SEED]
"""

import sys
import csv
import numpy as np
from nmdar_model import Popen

SEED = 20260101          # fixed seed (documented, for reproducibility)
N = 3000                 # Latin-hypercube sample size

# ---------------------------------------------------------------------------
# Parameter table (Table S-sens): name, canonical, low, high, dist, role
#   dist in {'unif','log'}; role in {'model','milieu'}
# ---------------------------------------------------------------------------
SPECS = [
    ('Kd_Glu',    3.0,   1.0,   5.0,   'log',  'model'),
    ('Kd_HCA',    20.0,  14.0,  56.0,  'log',  'model'),
    ('Kd_QUIN',   30.0,  10.0,  100.0, 'log',  'model'),
    ('Kd_Hcy_2A', 10.0,  5.0,   15.0,  'log',  'model'),
    ('Kd_Hcy_2B', 60.0,  40.0,  80.0,  'log',  'model'),
    ('Kd_Dser',   0.5,   0.2,   1.0,   'log',  'model'),
    ('Kd_Gly',    0.3,   0.1,   0.5,   'log',  'model'),
    ('eps_Hcy',   0.3,   0.2,   0.4,   'unif', 'model'),
    ('eps_QUIN',  1.0,   0.5,   1.0,   'unif', 'model'),
    ('n_g',       1.3,   1.0,   1.5,   'unif', 'model'),
    ('kappa',     1.0,   0.5,   2.0,   'unif', 'model'),
    ('Ki',        15.0,  5.0,   50.0,  'log',  'model'),
    ('Ksat',      0.30,  0.15,  0.50,  'unif', 'model'),
    ('w_2A',      0.6,   0.5,   0.7,   'unif', 'model'),
    ('G_2A',      0.805, 0.5635, 1.0465, 'unif', 'model'),   # +/- 30 %
    ('phi_infl',  0.7,   0.5,   0.9,   'unif', 'milieu'),
    ('HcyT_def',  25.0,  15.0,  30.0,  'unif', 'milieu'),
    ('Dser_high', 1.2,   0.8,   2.0,   'unif', 'milieu'),
    ('QUIN_high', 4.0,   2.0,   8.0,   'unif', 'milieu'),
]
NAMES = [s[0] for s in SPECS]
K = len(SPECS)
METRICS = ['O1', 'O2', 'O3', 'O4']


# ---------------------------------------------------------------------------
# Latin-hypercube sampling
# ---------------------------------------------------------------------------
def latin_hypercube(n, k, rng):
    """Return an (n, k) LHS design in [0, 1)."""
    u = np.empty((n, k))
    for j in range(k):
        perm = rng.permutation(n)
        u[:, j] = (perm + rng.random(n)) / n
    return u


def map_sample(u_row):
    """Map a [0,1)^K row to the named parameter values (dict)."""
    s = {}
    for val, (name, canon, lo, hi, dist, role) in zip(u_row, SPECS):
        if dist == 'log':
            s[name] = float(np.exp(np.log(lo) + val * (np.log(hi) - np.log(lo))))
        else:
            s[name] = float(lo + val * (hi - lo))
    return s


# ---------------------------------------------------------------------------
# Conditions and metrics for one sample
# ---------------------------------------------------------------------------
def model_params(s):
    """Extract the 15 model-parameter overrides from a sample dict."""
    return {name: s[name] for name, *_ , role in SPECS if role == 'model'}


def conditions(s):
    """Four milieu conditions for a sample (healthy baseline fixed)."""
    return {
        'Healthy':         dict(phi=0.1, HcyT=2, Dser=0.15, QUIN=0.2, Glu=1.0),
        'Inflammation':    dict(phi=s['phi_infl'], HcyT=2, Dser=s['Dser_high'],
                                QUIN=s['QUIN_high'], Glu=1.5),
        'MTHF_deficiency': dict(phi=0.1, HcyT=s['HcyT_def'], Dser=0.15,
                                QUIN=0.2, Glu=1.0),
        'Convergence':     dict(phi=s['phi_infl'], HcyT=s['HcyT_def'],
                                Dser=s['Dser_high'], QUIN=s['QUIN_high'], Glu=1.5),
    }


def crossover_dser(pr, lo=0.05, hi=5.0, iters=45):
    """[D-serine] at which the homocysteine effect changes sign (NaN if none).

    Hcy effect(d) = Popen(HcyT=30) - Popen(HcyT=1) in a low-oxidation context
    (phi=0.1, QUIN=0.2, Glu=1.0). Net-inhibitory (<0) at low D-serine, excitatory
    (>0) at high D-serine; the root is found by bisection.
    """
    def eff(d):
        return (Popen(0.1, 30.0, d, 0.2, 1.0, params=pr)
                - Popen(0.1, 1.0, d, 0.2, 1.0, params=pr))
    elo, ehi = eff(lo), eff(hi)
    if elo >= 0.0 or ehi <= 0.0:          # no sign change in physiological range
        return np.nan
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if eff(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def metrics(s):
    """Return (O1, O2, O3, O4) for a sample dict s."""
    pr = model_params(s)
    c = conditions(s)
    Ph = Popen(params=pr, **c['Healthy'])
    Pi = Popen(params=pr, **c['Inflammation'])
    Pd = Popen(params=pr, **c['MTHF_deficiency'])
    Pc = Popen(params=pr, **c['Convergence'])
    Padd = Ph + (Pi - Ph) + (Pd - Ph)
    O1 = Pc / Ph
    O2 = (Pc - Padd) / Padd
    O3 = Pd / Ph
    O4 = crossover_dser(pr)
    return O1, O2, O3, O4


# ---------------------------------------------------------------------------
# PRCC (precision-matrix method, manual matrix inverse — no LAPACK)
# ---------------------------------------------------------------------------
def _rank(x):
    return x.argsort().argsort().astype(float)


def _corr(R):
    """Correlation matrix of columns of R, BLAS-free (no np.corrcoef / gemm)."""
    Z = (R - R.mean(0)) / R.std(0)
    p = Z.shape[1]
    C = np.empty((p, p))
    for i in range(p):
        for j in range(i, p):
            c = float(np.mean(Z[:, i] * Z[:, j]))
            C[i, j] = C[j, i] = c
    return C


def _inv(A):
    """Gauss-Jordan inverse of a square matrix (pure numpy, no LAPACK)."""
    A = A.astype(float)
    n = A.shape[0]
    M = np.hstack([A, np.eye(n)])
    for i in range(n):
        p = i + int(np.argmax(np.abs(M[i:, i])))   # partial pivot
        if p != i:
            M[[i, p]] = M[[p, i]]
        M[i] = M[i] / M[i, i]
        for r in range(n):
            if r != i:
                M[r] = M[r] - M[r, i] * M[i]
    return M[:, n:]


def prcc(X, y):
    """PRCC of each column of X against y (rows with NaN y are dropped)."""
    ok = np.isfinite(y)
    Xc, yc = X[ok], y[ok]
    cols = [_rank(Xc[:, j]) for j in range(Xc.shape[1])] + [_rank(yc)]
    R = np.column_stack(cols)
    C = _corr(R)
    P = _inv(C)
    out = Xc.shape[1]
    return np.array([-P[i, out] / np.sqrt(P[i, i] * P[out, out])
                     for i in range(out)])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    U = latin_hypercube(n, K, rng)
    X = np.empty((n, K))
    O = np.empty((n, 4))
    for r in range(n):
        s = map_sample(U[r])
        X[r] = [s[name] for name in NAMES]
        O[r] = metrics(s)

    O1, O2, O3, O4 = O[:, 0], O[:, 1], O[:, 2], O[:, 3]

    # --- claim fractions ---
    frac = {
        'O1 (fold>1)':       np.mean(O1 > 1.0),
        'O2 (excess>0)':     np.mean(O2 > 0.0),
        'O3 (fold<=1)':      np.mean(O3 <= 1.0),
        'O4 (crossover)':    np.mean(np.isfinite(O4)),
    }
    print(f"N = {n} | seed = {seed}\n")
    print("Claim fractions (held across the ensemble):")
    for k, v in frac.items():
        print(f"  {k:20s} {100*v:6.1f} %")

    # --- medians + 5-95 percentiles ---
    def q(a):
        a = a[np.isfinite(a)]
        return np.median(a), np.percentile(a, 5), np.percentile(a, 95)
    print("\nMetric distributions (median [5-95 %]):")
    labels = {'O1': 'convergence fold', 'O2': 'supra-additive excess',
              'O3': 'deficiency fold', 'O4': 'D-serine crossover (uM)'}
    stats = {}
    for name, a in zip(METRICS, [O1, O2, O3, O4]):
        m, lo, hi = q(a)
        stats[name] = (m, lo, hi)
        if name == 'O2':
            print(f"  {name} {labels[name]:24s} {100*m:+6.1f}% "
                  f"[{100*lo:+.1f}%, {100*hi:+.1f}%]")
        else:
            print(f"  {name} {labels[name]:24s} {m:6.3f} [{lo:.3f}, {hi:.3f}]")

    # --- PRCC ---
    PR = np.column_stack([prcc(X, O[:, j]) for j in range(4)])
    print("\nTop PRCC drivers per metric:")
    for j, name in enumerate(METRICS):
        order = np.argsort(-np.abs(PR[:, j]))[:3]
        tops = ", ".join(f"{NAMES[i]} {PR[i, j]:+.2f}" for i in order)
        print(f"  {name}: {tops}")

    # --- write CSVs ---
    with open('sens_lhs_outputs.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(NAMES + METRICS)
        for r in range(n):
            w.writerow([f"{X[r, j]:.6g}" for j in range(K)] +
                       [f"{O[r, j]:.6g}" for j in range(4)])
    with open('sens_prcc.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['parameter'] + METRICS)
        for i in range(K):
            w.writerow([NAMES[i]] + [f"{PR[i, j]:+.4f}" for j in range(4)])
    with open('sens_summary.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'claim', 'fraction_%', 'median', 'pct5', 'pct95'])
        claim = {'O1': 'fold>1', 'O2': 'excess>0', 'O3': 'fold<=1',
                 'O4': 'crossover exists'}
        fr = {'O1': frac['O1 (fold>1)'], 'O2': frac['O2 (excess>0)'],
              'O3': frac['O3 (fold<=1)'], 'O4': frac['O4 (crossover)']}
        for name in METRICS:
            m, lo, hi = stats[name]
            w.writerow([name, claim[name], f"{100*fr[name]:.1f}",
                        f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    print("\nwrote sens_lhs_outputs.csv, sens_prcc.csv, sens_summary.csv")
    return X, O, PR


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else SEED
    main(n, seed)
