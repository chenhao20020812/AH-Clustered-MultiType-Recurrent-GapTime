# AH-Clustered-MultiType-Recurrent-GapTime

MATLAB implementation of accelerated hazards models for clustered multi-type recurrent gap-time data.

This repository provides MATLAB code for a semiparametric accelerated hazards model for clustered recurrent gap-time data with multiple event types. The implementation includes simulation studies, weighted estimating equations, multiplier resampling inference, baseline cumulative hazard estimation, and real-data applications.

---

## Overview

Recurrent event data frequently arise in medical studies and longitudinal follow-up studies, where subjects may experience multiple events over time. These data may contain multiple event types and within-cluster dependence.

This repository implements an accelerated hazards (AH) model for clustered multi-type recurrent gap-time data. The framework allows:

- clustered recurrent gap-time observations;
- multiple event types with type-specific baseline hazard functions;
- covariate effects through an accelerated time scale;
- weighted estimating equation estimation;
- multiplier resampling-based variance estimation.

The accelerated hazards model is defined as

```math
\lambda_{ijkr}(t|Z_{ikr})
=
\lambda_{0r}(t\exp(\beta^T Z_{ikr}))
where $r$ denotes the event type and each event type has its own baseline hazard function.
Repository Structure

The repository is organized as follows:
AH-Clustered-MultiType-Recurrent-GapTime/

├── Simulation/
│   ├── Data generation
│   ├── Parameter estimation
│   └── Simulation experiments
│
├── Application_MIMIC/
│   ├── MIMIC-IV real-data application
│   └── Baseline cumulative hazard estimation
│
├── Functions/
│   ├── Regression parameter estimation
│   ├── Score functions
│   ├── Variance estimation
│   └── Auxiliary functions
│
├── Figures/
│
├── README.md
│
└── LICENSE
