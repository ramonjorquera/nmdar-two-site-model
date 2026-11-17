# -*- coding: utf-8 -*-
"""
Clean per-condition error bars for the 2x2 bar/point graph.

Averages MANY independent two-state-channel traces (N=50) so the Monte-Carlo
mean lands on the deterministic P_open and the SD is stable. Same CTMC method
and tau_c as nmdar_noise_layer.py (step factor k = 1 - exp(-dt/tau_c)).

Reports, per condition:
  P_deterministic          -> bar height (exact ensemble mean; use THIS as the point)
  MC_mean_check            -> sanity: ~= P_deterministic (should match to ~1e-3)
  SD_window100ms_N50       -> DE_ruido: SD of a 100 ms-window-averaged measurement
                              (matches block-averaging into 100 ms blocks) -> the error bar
  SEM_window100ms          -> SD_window / sqrt(n_blocks) IF you averaged n_blocks=10 (1000/100)
  SD_instant_N50           -> instantaneous open-fraction spread (noise-band width)
  analytic_SD_window100ms  -> closed-form check sqrt(P(1-P)/N * g(100ms))

BLAS-free (elementwise numpy + RNG only). Reproducible seed.
Run: C:/Users/MonraDJ/miniforge3/envs/md/python.exe nmdar_bar_error_N50.py
"""
import numpy as np
from nmdar_model import Popen, CONDITIONS

SEED, tau_c, N = 20261109, 50.0, 50
rng = np.random.default_rng(SEED)
ORDER = ['Healthy', 'Inflammation', 'MTHF_deficiency', 'Convergence']
P = {k: Popen(**c) for k, c in CONDITIONS.items()}

T_win, dt = 100.0, 0.5          # 100 ms window, 0.5 ms step  -> 200 steps/window
n_windows = 8000                # many independent 100 ms windows -> stable mean & SD
steps = int(round(T_win/dt))
k = 1.0 - np.exp(-dt/tau_c)     # exact two-state step factor

def g(T, tc=tau_c):
    r = tc/T
    return 2.0*(r - r*r*(1.0 - np.exp(-T/tc)))

def run(Pv):
    """Return (window_means, inst_sum, inst_sumsq, inst_count) for n_windows traces."""
    st = rng.random((n_windows, N)) < Pv        # init at steady state
    acc = np.zeros(n_windows)                    # time-sum of open fraction per window
    isum = 0.0; isq = 0.0; icnt = 0              # pooled instantaneous stats
    for _ in range(steps):
        u = rng.random((n_windows, N))
        opens  = (~st) & (u < Pv*k)
        closes = st & (u < (1.0-Pv)*k)
        st = (st | opens) & ~closes
        frac = st.mean(axis=1)                   # instantaneous open fraction per window
        acc += frac
        isum += frac.sum(); isq += (frac*frac).sum(); icnt += frac.size
    wmean = acc/steps                            # 100 ms-window-averaged open fraction
    inst_mean = isum/icnt
    inst_sd = np.sqrt(max(isq/icnt - inst_mean*inst_mean, 0.0))
    return wmean, inst_sd

rows = []
print(f"N={N} channels, tau_c={tau_c} ms, window={T_win} ms, {n_windows} windows, seed={SEED}\n")
print(f"{'condition':16s} {'P_det':>7s} {'MCmean':>7s} {'SDwin':>7s} {'analSD':>7s} {'SDinst':>7s}")
for cond in ORDER:
    Pv = P[cond]
    wmean, inst_sd = run(Pv)
    mc_mean = wmean.mean()
    sd_win = wmean.std(ddof=1)
    sem10 = sd_win/np.sqrt(10.0)                 # if you average 10 blocks (1000/100 ms)
    an_sd = np.sqrt(Pv*(1.0-Pv)/N * g(T_win))
    rows.append((cond, Pv, mc_mean, sd_win, sem10, inst_sd, an_sd))
    print(f"{cond:16s} {Pv:7.4f} {mc_mean:7.4f} {sd_win:7.4f} {an_sd:7.4f} {inst_sd:7.4f}")

# ---- clean wide CSV for Origin (point = P_deterministic, error = SD_window100ms_N50) ----
out = 'bar_Popen_error_N50.csv'
with open(out, 'w', encoding='utf-8') as f:
    f.write('condition,P_deterministic,MC_mean_check,SD_window100ms_N50,'
            'SEM_window100ms_10blocks,SD_instant_N50,analytic_SD_window100ms\n')
    for cond, Pv, mc, sdw, sem, sdi, an in rows:
        f.write(f'{cond},{Pv:.4f},{mc:.4f},{sdw:.4f},{sem:.4f},{sdi:.4f},{an:.4f}\n')
print(f"\nwrote {out}")
print("Origin: plot column P_deterministic vs condition; Y-error = SD_window100ms_N50.")
