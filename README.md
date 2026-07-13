# AH-Clustered-MultiType-Recurrent-GapTime

MATLAB implementation of an accelerated hazards model for clustered multi-type recurrent gap-time data.

This repository contains code for simulation studies and a real-data application of a semiparametric accelerated hazards model with multiple event types. The method uses weighted estimating equations, multiplier resampling inference, and type-specific baseline cumulative hazard estimation.

---

## Model

The accelerated hazards model is defined as:

$$
\lambda_{ijkr}(t|Z_{ikr})
=
\lambda_{0r}(t\exp(\beta^T Z_{ikr}))
$$

where $r$ denotes the event type and each event type has its own baseline hazard function.

---

## Repository Structure


AH-Clustered-MultiType-Recurrent-GapTime/

├── Simulation/
│ └── Simulation studies
│
├── Application_MIMIC/
│ └── MIMIC-IV real-data analysis
│
├── Functions/
│ └── Model estimation and inference functions
│
├── README.md
└── LICENSE


---

## Requirements

- MATLAB R2022a or later
- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox (for multiplier resampling)

---

## Main Functions

### Parameter estimation


estimate_beta_AH_multi.m
score_AH_multi.m


Estimate regression parameters through weighted estimating equations.

### Baseline hazard estimation


baseline_AH_multi.m


Estimate type-specific baseline cumulative hazard functions.

### Variance estimation


var_AH_resample_multi.m
var_AH_resample_multi1.m


Obtain variance estimates using multiplier resampling.

---

## Simulation

Simulation codes are provided in:


Simulation/


The simulation studies evaluate parameter estimation performance, including bias, standard error, and coverage probability.

---

## Real Data Application

A MIMIC-IV application is provided in:


Application_MIMIC/


The analysis considers recurrent hospitalization events with multiple event types.

The raw MIMIC-IV data are not included due to data access restrictions.

---

## Citation

If you use this code, please cite:

Chen, Hao.

*An Accelerated Hazards Model for Clustered Multi-Type Recurrent Gap-Time Data.*

---

## License

This project is released under the MIT License.
