# Claim-Relative Evidence Quality and Support

Reproducibility package for the paper:

> **Claim-Relative Evidence Quality and Support in AI-Augmented Legacy Modernization: A Formal Representation Model with Aggregation Properties**
>
> Fady Nashat Manhary
> Faculty of Computers and Information, Assiut University

---

## Overview

This repository contains the reproducibility package for the paper. It includes:

- **Four aggregation models** implemented in Python:
  - `M0`: unweighted baseline (equal weights for all evidence)
  - `M1`: raw-weight baseline (geometric-mean quality weight, no procedural filter)
  - `M1'`: procedurally filtered baseline (excludes non-`ACCEPTED` evidence)
  - `M2`: full model (procedural exclusion + numerical eligibility gate)
- **Nine controlled test cases** (A–I) reported in Appendix A of the paper.
- **Automated checks** for:
  - input validation (including `NaN` and `±Infinity`)
  - raw weights
  - numerical eligibility (including boundary values)
  - classification thresholds (inclusive)
  - aggregate outputs
  - removal-sensitivity bounds

The package does **not** establish empirical effectiveness. It verifies that the implementation is consistent with the definitions in the paper and that the published numerical values are reproducible.

---

## Repository Structure
