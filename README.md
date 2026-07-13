AH-Clustered-MultiType-Recurrent-GapTime

MATLAB implementation of accelerated hazards models for clustered multi-type recurrent gap-time data.

This repository provides MATLAB code for a semiparametric accelerated hazards model for clustered recurrent gap-time data with multiple event types. The implementation includes simulation studies, weighted estimating equations, multiplier resampling inference, baseline cumulative hazard estimation, and real-data applications.

Overview

Recurrent event data are frequently observed in medical studies, reliability studies, and other longitudinal applications. In many situations, subjects may experience multiple recurrent events, different event types, and within-cluster dependence.

This repository implements an accelerated hazards (AH) model for clustered multi-type recurrent gap-time data. The proposed framework allows:

clustered recurrent gap-time observations;
multiple event types with type-specific baseline hazard functions;
covariate effects through an accelerated time scale;
weighted estimating equation estimation;
multiplier resampling-based variance estimation.

The accelerated hazards model is defined as

λ
ijkr
	​

(t∣Z
ikr
	​

)=λ
0r
	​

(texp(β
T
Z
ikr
	​

)),

where $r$ represents the event type and each event type has its own baseline hazard function.

Repository Structure
AH-Clustered-MultiType-Recurrent-GapTime/

├── Simulation/
│   ├── Simulation data generation
│   ├── Parameter estimation
│   └── Simulation experiments
│
├── Application_MIMIC/
│   ├── MIMIC-IV real-data application
│   └── Baseline hazard estimation and confidence intervals
│
├── Functions/
│   ├── Estimation functions
│   ├── Score functions
│   ├── Variance estimation
│   └── Auxiliary functions
│
├── README.md
│
└── LICENSE
Requirements

The code was developed using:

MATLAB R2022a or later

Recommended MATLAB toolboxes:

Statistics and Machine Learning Toolbox
Parallel Computing Toolbox (for parallel multiplier resampling)
Main Functions
Parameter Estimation

The regression coefficient is estimated by solving weighted estimating equations:

U(β)=0.

Main functions:

estimate_beta_AH_multi.m

score_AH_multi.m

These functions implement estimation of regression parameters for clustered recurrent gap-time data with multiple event types.

Baseline Cumulative Hazard Estimation

For each event type, the baseline cumulative hazard function is estimated by:

Λ
0r
	​

(t)=∫
0
t
	​

λ
0r
	​

(u)du.

Main function:

baseline_AH_multi.m

The method allows each event type to have its own baseline cumulative hazard function.

Variance Estimation

The variance of the estimator is obtained using multiplier resampling.

Main functions:

var_AH_resample_multi.m

var_AH_resample_multi1.m

The parallel version uses MATLAB parfor to improve computational efficiency.

Simulation Studies

Simulation studies are conducted to evaluate the finite-sample performance of the proposed estimator.

The simulation investigates:

Bias;
Empirical standard error;
Estimated standard error;
Coverage probability;
Baseline cumulative hazard estimation.

Simulation-related codes are provided in:

Simulation/
Real Data Application

A real-data application based on the MIMIC-IV database is included.

The analysis considers recurrent hospitalization events with multiple event types.

The event types include:

Circulatory diseases;
Respiratory diseases;
Infectious diseases.

Main application script:

Application_MIMIC/

application_MIMIC_AH_type123_baseline_CI.m
Data Availability

The MIMIC-IV dataset used in the real-data application is publicly available through PhysioNet:

https://physionet.org/content/mimiciv/

Access requires:

a PhysioNet account;
completion of the required training;
approval of data access credentials.

The raw MIMIC-IV data are not included in this repository due to data usage restrictions.

How to Run
Simulation
Add all folders to the MATLAB path.
Run:
main.m
MIMIC-IV Application
Download and preprocess the MIMIC-IV dataset.
Modify the data path in:
application_MIMIC_AH_type123_baseline_CI.m
Run the MATLAB script.
Citation

If you use this code, please cite:

Chen, Hao.

An Accelerated Hazards Model for Clustered Multi-Type Recurrent Gap-Time Data.

License

This project is released under the MIT License.

See the LICENSE file for details.
