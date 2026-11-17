# NMDAR two-site convergent-pharmacology model — companion code

Reproducibility code for:

> Jorquera, R. A. & Stehberg, J. *Convergent pharmacology at the two agonist
> sites of the NMDA receptor: homocysteine, quinolinic acid and co-agonist tone
> as a permissive amplifier of excitotoxic signaling.*

A minimal two-site occupancy × efficacy model of the NMDA receptor that computes
the steady-state ambient open probability (P_open) of a forebrain GluN2A/2B pool
as a function of the metabolic–inflammatory milieu. The scripts reproduce every
numerical result in the paper: the central conditions (Table 1), the global and
local sensitivity analyses (Section 3.6), the Figure 2 panels, the stochastic
channel-noise layer (Section 3.7), the Figure 4D error bars, and the regional
paired-pulse-ratio correlation (Supplementary).

## Requirements

- Python 3 and **numpy** only.
- No `scipy`, no LAPACK, no BLAS: the PRCC matrix inverse and the correlation
  matrices are implemented in pure elementwise numpy, so the code runs on
  restricted environments. Analyses with a random component use a fixed,
  documented seed.

## Files

| Script | Reproduces | Key outputs |
|---|---|---|
| `nmdar_model.py` | The model and Table 1 (P_open and supra-additivity of the convergent condition) | `table1_popen.csv` |
| `nmdar_sensitivity_lhs.py` | §3.6 global sensitivity: Latin-hypercube ensemble (N = 3000), claim fractions, PRCC | `sens_lhs_outputs.csv`, `sens_prcc.csv`, `sens_summary.csv` |
| `nmdar_sensitivity_oat.py` | §3.6 local sensitivity: ±50 % tornado on the convergence fold and the crossover-vs-Ki map | `sens_tornado.csv`, `sens_crossover_vs_Ki.csv` |
| `nmdar_figure2_data.py` | Figure 2 (A crossover, B Hcy×D-serine gate, C GluN2A/2B homocysteine sensitivity, D Hcy×φ oxidation) | `fig2B_*`, `fig2C_*`, `subunit_hcy_sensitivity.csv`, `fig2phi_*`, `phi_hcy_panel.csv` |
| `nmdar_noise_layer.py` | §3.7 stochastic two-state channel-gating layer: discriminability d′, Monte-Carlo validation, example traces | `noise_discriminability.csv`, `noise_traces.csv`, `noise_meanvar_check.csv`, `noise_condition_meanSD.csv` |
| `nmdar_bar_error_N50.py` | Figure 4D channel-noise error bars | `bar_Popen_error_N50.csv` |
| `ppr_correlation.py` | Supplementary Fig S-PPR / Table S-reg: regional model-vs-literature paired-pulse ratio | `PPR_correlation.csv` |

`nmdar_model.py` is the single source of the model math; every other script
imports it. Figure panels are drawn in OriginPro from the CSV / `.dat` outputs.

## Usage

```bash
python nmdar_model.py            # Table 1 + supra-additivity
python nmdar_sensitivity_lhs.py  # global sensitivity (N=3000 by default)
python nmdar_sensitivity_oat.py  # tornado + crossover-vs-Ki
python nmdar_figure2_data.py     # Figure 2 panel data
python nmdar_noise_layer.py      # stochastic layer (d', traces)
python nmdar_bar_error_N50.py    # Figure 4D error bars
python ppr_correlation.py        # regional PPR correlation
```

`nmdar_sensitivity_lhs.py` accepts optional `N` and `SEED` arguments
(`python nmdar_sensitivity_lhs.py 3000 20260101`).

## Model, in brief

Glutamate-site ligands (glutamate, homocysteic acid, quinolinic acid,
homocysteine) contribute a normalized drive `f_i = [L_i]/Kd_i` and an intrinsic
efficacy `eps_i`; homocysteine is a weak, self-gated partial agonist (`eps = 0.3`)
that also antagonizes the co-agonist (GluN1) site. The glutamate site is
cooperative (Hill `n_g`) and negatively coupled (`kappa`) to the co-agonist site,
solved self-consistently; a redox fraction `phi` partitions total homocysteine
between the weak parent and the potent acid (`HCA = 1 + phi·HcyT`). P_open sums the
GluN2A and GluN2B pools weighted by occupancy and single-channel conductance.
Parameters and provenance are in `nmdar_model.py` (`DEFAULT_PARAMS`) and in the
manuscript (Section 2, Table S-sens).

## Data

The accompanying `cal_refined_*` CSV files hold the Figure 3 dynamics
(current-clamp Vm trains, isolated AMPA/NMDA currents) and the Supplementary
voltage-layer and regional tables used by the Origin figures.

## License

MIT — see `LICENSE`.
