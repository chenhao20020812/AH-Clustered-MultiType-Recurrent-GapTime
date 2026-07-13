# AH-Clustered-MultiType-Recurrent-GapTime

MATLAB implementation of an accelerated hazards model for clustered multi-type recurrent gap-time data.

This repository contains MATLAB code for simulation studies and a real-data application of a semiparametric accelerated hazards model with multiple event types. The method uses weighted estimating equations, multiplier resampling inference, and type-specific baseline cumulative hazard estimation.

---

## Overview

Recurrent event data frequently arise in medical studies and longitudinal follow-up studies. Subjects may experience multiple recurrent events, different event types, and within-cluster dependence.

This repository implements an accelerated hazards (AH) model for clustered multi-type recurrent gap-time data. The framework includes:

- clustered recurrent gap-time observations;
- multiple event types with type-specific baseline hazard functions;
- covariate effects through an accelerated time scale;
- weighted estimating equation estimation;
- multiplier resampling-based variance estimation.

The model can be written as:

lambda_ijkr(t | Z_ikr) = lambda_0r(t exp(beta^T Z_ikr))

where r denotes the event type and each event type has its own baseline hazard function.

---

## Repository Structure

The repository is organized as follows:


AH-Clustered-MultiType-Recurrent-GapTime/

Simulation/
Simulation studies

Application_MIMIC/
MIMIC-IV real-data analysis

Functions/
Model estimation and inference functions

README.md

LICENSE


---

## Requirements

The code was developed using:

- MATLAB R2022a or later

Recommended MATLAB toolboxes:

- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox

---

## Main Functions

### Parameter Estimation

The regression parameters are estimated using weighted estimating equations.

Main functions:


estimate_beta_AH_multi.m

score_AH_multi.m


---

### Baseline Hazard Estimation

Type-specific baseline cumulative hazard functions are estimated using:


baseline_AH_multi.m


---

### Variance Estimation

Multiplier resampling is used for variance estimation.

Main functions:


var_AH_resample_multi.m

var_AH_resample_multi1.m


Parallel computation is implemented using MATLAB `parfor`.

---

## Simulation Studies

Simulation codes are provided in:


Simulation/


The simulation studies evaluate:

- Bias;
- Empirical standard error;
- Estimated standard error;
- Coverage probability;
- Baseline cumulative hazard estimation.

---

## Real Data Application

A real-data application based on the MIMIC-IV database is included.

The analysis considers recurrent hospitalization events with multiple event types.

The MIMIC-IV data are not included in this repository due to data access restrictions.

Application codes are provided in:


Application_MIMIC/


---

## Data Availability

The MIMIC-IV dataset is publicly available through PhysioNet:

https://physionet.org/content/mimiciv/

Access requires completion of the required training and approval of data access credentials.

---

## How to Run

### Simulation

Add all folders to the MATLAB path and run the simulation script:


main.m


### MIMIC-IV Application

Download and preprocess the MIMIC-IV dataset, then modify the data path in the application script:


application_MIMIC_AH_type123_baseline_CI.m


---

## Citation

If you use this code, please cite:

Chen, Hao.

*An Accelerated Hazards Model for Clustered Multi-Type Recurrent Gap-Time Data.*

---

## License

This project is released under the MIT License.
