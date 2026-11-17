# -*- coding: utf-8 -*-
"""
Stochastic (channel-noise) layer for the NMDAR two-site model.
Message for BJP: the deterministic ambient P_open is the ENSEMBLE MEAN of a
stochastic two-state gating process, and the ~2.6x convergence signal is
distinguishable from the healthy state ABOVE channel / trial-to-trial noise
for physiologically realistic receptor numbers.

Each receptor is a two-state (closed<->open) continuous-time Markov channel with
steady-state open probability = the model's P_open for that condition and a gating
correlation time tau_c. Whole-cell signal = number of open channels (arbitrary unit).

Analytic variance of the time-averaged open fraction over a window T, for N
independent channels (exact for the two-state CTMC):
    Var = [P(1-P)/N] * g(T),  g(T) = 2[(tau_c/T) - (tau_c/T)^2 (1 - e^{-T/tau_c})]
Discriminability convergence-vs-healthy: d' = (P_conv - P_health)/sqrt(0.5(Var_c+Var_h)).

Monte-Carlo validates the analytic SD and that MC mean = deterministic P_open.
BLAS-free (only elementwise numpy + RNG). Reproducible seed.
Run: C:/Users/MonraDJ/miniforge3/envs/md/python.exe nmdar_noise_layer.py
"""
import numpy as np
from nmdar_model import Popen, CONDITIONS

SEED   = 20261109
tau_c  = 50.0      # ms, NMDAR gating correlation time (illustrative; larger = more conservative)
rng    = np.random.default_rng(SEED)

# deterministic ambient P_open per condition (the ensemble mean)
P = {name: Popen(**c) for name, c in CONDITIONS.items()}
ORDER = ['Healthy', 'Inflammation', 'MTHF_deficiency', 'Convergence']
print("deterministic P_open:", {k: round(P[k], 4) for k in ORDER})

def g(T, tc=tau_c):
    r = tc/T
    return 2.0*(r - r*r*(1.0 - np.exp(-T/tc)))

def var_mean(Pv, N, T):
    return Pv*(1.0-Pv)/N * g(T)

# ------------------------------------------------------------------ #
#  (1) Discriminability d'(N) at three averaging windows (analytic)  #
# ------------------------------------------------------------------ #
Ns = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
windows = [100.0, 200.0, 500.0]   # ms
Ph, Pc = P['Healthy'], P['Convergence']
with open('noise_discriminability.csv', 'w', encoding='utf-8') as f:
    f.write('N,' + ','.join(f'dprime_T{int(T)}ms' for T in windows) + '\n')
    for N in Ns:
        dps = []
        for T in windows:
            vh, vc = var_mean(Ph, N, T), var_mean(Pc, N, T)
            dps.append((Pc - Ph)/np.sqrt(0.5*(vh+vc)))
        f.write(f'{N},' + ','.join(f'{d:.4f}' for d in dps) + '\n')

# ------------------------------------------------------------------ #
#  (2) Per-condition mean +/- SD of the time-averaged open fraction  #
#      vs N (T = 200 ms) -> 4 clouds that overlap at low N, separate #
# ------------------------------------------------------------------ #
Tref = 200.0
with open('noise_condition_meanSD.csv', 'w', encoding='utf-8') as f:
    f.write('N,' + ','.join(f'mean_{k},sd_{k}' for k in ORDER) + '\n')
    for N in Ns:
        cells = []
        for k in ORDER:
            cells.append(f'{P[k]:.4f}')
            cells.append(f'{np.sqrt(var_mean(P[k], N, Tref)):.4f}')
        f.write(f'{N},' + ','.join(cells) + '\n')

# ------------------------------------------------------------------ #
#  (3) Monte-Carlo validation (two-state CTMC, exact step probs)     #
# ------------------------------------------------------------------ #
def simulate(Pv, N, trials, T, dt):
    steps = int(round(T/dt))
    k = 1.0 - np.exp(-dt/tau_c)          # exact two-state step factor
    st = rng.random((trials, N)) < Pv    # init at steady state
    acc = np.zeros(trials)
    for _ in range(steps):
        u = rng.random((trials, N))
        # closed->open w.p. Pv*k ; open->closed w.p. (1-Pv)*k
        opens  = (~st) & (u < Pv*k)
        closes = st & (u < (1.0-Pv)*k)
        st = (st | opens) & ~closes
        acc += st.mean(axis=1)
    return acc/steps                      # time-averaged open fraction per trial

Nmc, trials, Tmc, dt = 50, 2000, Tref, 0.5
with open('noise_meanvar_check.csv', 'w', encoding='utf-8') as f:
    f.write('condition,P_det,MC_mean,MC_sd,analytic_sd,CV_percent\n')
    print(f"\nMC validation (N={Nmc}, trials={trials}, T={Tmc} ms, tau_c={tau_c} ms):")
    for k in ORDER:
        samp = simulate(P[k], Nmc, trials, Tmc, dt)
        asd = np.sqrt(var_mean(P[k], Nmc, Tmc))
        cv = 100.0*samp.std()/samp.mean()
        f.write(f'{k},{P[k]:.4f},{samp.mean():.4f},{samp.std():.4f},{asd:.4f},{cv:.1f}\n')
        print(f"  {k:16s} P={P[k]:.4f}  MCmean={samp.mean():.4f}  MCsd={samp.std():.4f}"
              f"  (analytic {asd:.4f})  CV={cv:.0f}%")

# ------------------------------------------------------------------ #
#  (4) Example noisy whole-cell traces (N=50, one trial per cond)    #
# ------------------------------------------------------------------ #
def trace(Pv, N, T, dt):
    steps = int(round(T/dt)); k = 1.0 - np.exp(-dt/tau_c)
    st = rng.random(N) < Pv; out = np.empty(steps)
    for i in range(steps):
        u = rng.random(N)
        st = (st | ((~st) & (u < Pv*k))) & ~(st & (u < (1.0-Pv)*k))
        out[i] = st.sum()
    return out

Ttr, dttr, Ntr = 1000.0, 0.5, 50
steps = int(round(Ttr/dttr))
t = np.arange(steps)*dttr
traces = {k: trace(P[k], Ntr, Ttr, dttr) for k in ORDER}
ds = max(1, int(round(1.0/dttr)))          # downsample to ~1 ms rows
with open('noise_traces.csv', 'w', encoding='utf-8') as f:
    f.write('time_ms,' + ','.join(ORDER) + f'   (N={Ntr} channels, open-channel count)\n')
    for i in range(0, steps, ds):
        f.write(f'{t[i]:.1f},' + ','.join(f'{int(traces[k][i])}' for k in ORDER) + '\n')

# ------------------------------------------------------------------ #
with open('noise_summary.txt', 'w', encoding='utf-8') as f:
    def w(s): f.write(s+'\n'); print(s)
    w(f"\nSTOCHASTIC CHANNEL-NOISE LAYER  (tau_c={tau_c} ms, seed={SEED})")
    w(f"deterministic ambient P_open = ensemble mean: " +
      ", ".join(f"{k} {P[k]:.3f}" for k in ORDER))
    dN = {int(N): (Pc-Ph)/np.sqrt(0.5*(var_mean(Ph,N,Tref)+var_mean(Pc,N,Tref))) for N in Ns}
    w(f"Convergence-vs-Healthy d' (T={int(Tref)} ms): "
      f"N=1 -> {dN[1]:.2f}, N=10 -> {dN[10]:.2f}, N=50 -> {dN[50]:.2f}, "
      f"N=100 -> {dN[100]:.2f}, N=500 -> {dN[500]:.2f}")
    # N at which d'>=1 (reliably separable) and >=2
    def Ncrit(thr):
        for N in range(1, 2000):
            if (Pc-Ph)/np.sqrt(0.5*(var_mean(Ph,N,Tref)+var_mean(Pc,N,Tref))) >= thr:
                return N
        return None
    w(f"  separable (d'>=1) from N={Ncrit(1)} channels; well-separated (d'>=2) from N={Ncrit(2)} "
      f"(200 ms window).")
    w("wrote: noise_discriminability.csv, noise_condition_meanSD.csv, "
      "noise_meanvar_check.csv, noise_traces.csv")

print("\ndone.")
