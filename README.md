# AH-Clustered-MultiType-Recurrent-GapTime

MATLAB implementation of accelerated hazards models for clustered multi-type recurrent gap-time data.

This repository provides MATLAB code for a semiparametric accelerated hazards model designed for clustered recurrent gap-time data with multiple event types. The implementation includes simulation studies, weighted estimating equations, multiplier resampling inference, baseline cumulative hazard estimation, and real-data applications.

---

## Overview

Recurrent event data frequently arise in medical and reliability studies, where subjects may experience multiple events over time. In many applications, recurrent events may belong to different types and observations may exhibit clustering structures.

This repository implements an accelerated hazards (AH) modeling framework that allows:

- clustered recurrent gap-time observations;
- multiple event types with type-specific baseline hazard functions;
- covariate effects through an accelerated time scale;
- weighted estimating equation estimation;
- multiplier resampling-based variance estimation.

The model is formulated as

\[
\lambda_{ijkr}(t|Z_{ikr})
=
\lambda_{0r}\left(t\exp(\beta^T Z_{ikr})\right),
\]

where \(r\) denotes the event type and each event type has its own baseline hazard function.

---

# Repository Structure
