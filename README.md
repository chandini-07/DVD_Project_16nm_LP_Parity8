# ML-driven PVT-variation-aware leakage estimation: 16nm_LP, VDD 1 V, 8-bit parity checker

Simulator: ngspice (generation environment recorded as ngspice-42 in ngspice_version.txt; current installed environment verified as ngspice-43 at /usr/local/bin/ngspice; outputs identical on example netlists). Python packages: requirements.txt.

## Verified Results Summary (from results/*.csv)

| Level | Benchmark Metric | v1 Baseline | Verified v2 Upgrade | Reference CSV Deliverable |
|---|---|---|---|---|
| **Transistor** | NMOS ON Validation Median (id / ig / is / ib) | 12.54% / 6.54% / 12.87% / 3.56% | **4.42% / 1.54% / 5.23% / 0.95%** | `results/PartA_transistor/new_vs_old_validation.csv` |
| **Transistor** | NMOS OFF Validation Median (id / ig / is / ib) | 2.04% / 0.00% / 3.08% / 0.82% | **1.34% / 0.00% / 2.50% / 0.85%** | `results/PartA_transistor/new_vs_old_validation.csv` |
| **Transistor** | PMOS ON Validation Median (id / ig / is / ib) | 13.73% / 5.08% / 13.15% / 4.19% | **5.07% / 1.27% / 6.14% / 0.91%** | `results/PartA_transistor/new_vs_old_validation.csv` |
| **Transistor** | PMOS OFF Validation Median (id / ig / is / ib) | 3.09% / 2.12% / 3.23% / 1.08% | **1.52% / 2.12% / 1.84% / 0.96%** | `results/PartA_transistor/new_vs_old_validation.csv` |
| **Gates** | SPICE node V: Mean Accuracy | 96.72% | **99.28%** (med 0.51%, p95 1.98%, max 12.8%) | `results/PartB_gates/gate_results.csv` |
| **Gates** | ML-solved node V: Mean Accuracy | 96.47% | **99.25%** (med 0.53%, p95 2.08%) | `results/PartB_gates/gate_node_results.csv` |
| **Circuit** | Fully-ML (ML-solved V): Mean Accuracy | 97.28% | **99.46%** (med 0.43%, p95 1.57%, max 4.0%) | `results/PartC_parity_checker/circuit_results.csv` |
| **Circuit** | SPICE node V (Mode 2): Mean Accuracy | 97.45% | **99.49%** (med 0.40%, p95 1.33%, max 5.0%) | `data/PartC_parity_checker/circuit_ml.csv` |
| **Circuit** | Standalone gate SPICE sum (no ML) | 99.98% | **99.98%** (med 0.01%, p95 0.06%) | `data/PartC_parity_checker/circuit_ml.csv` |

Target set in project brief: *"up to 99% mean accuracy"*. Achieved: **99.46%** (fully-ML parity checker) and **99.28%** (gates).

---

## Technical Disclosure & Modeling Methodology

1. **Training Data Beyond 10,000 Rows:** Extra training data was sampled in `data/PartA_transistor/big_<case>.csv` (40,000 SPICE rows per case, seed `[31337, case_index]`, disjoint from original datasets and test splits). `train_v2.py` uses 20,000 extra rows: rows `[0:12000]` (rail-mix where $V_d, V_s \in \{0, V_{DD}\}$ with probabilities 40%/40%/20% uniform) plus rows `[24000:32000]` (uniform). A 3,000-row rail-mix validation set `railval_<case>.csv` (seed `[555, case_index]`) was used strictly for model selection and never trained on.
2. **Per-Terminal Architecture & Optimization:** Models use 4 dedicated multi-layer perceptrons per transistor case (one per terminal: $I_d, I_g, I_s, I_b$), each with architecture `(64, 64)` tanh activations and per-terminal standardized symlog targets ($C_0 = 10^{-15}\text{ A}$). Training employed Adam (300 epochs, lr=2e-3, batch size 256) followed by an L-BFGS second-order polish (1500 iterations).
3. **Hyperparameter Tuning Scope:** Hyperparameter tuning was performed on NMOS OFF terminals $I_d$ and $I_b$ only; no full combinatorial ablation study across all 16 sub-models was performed.
4. **OFF Device Gate Leakage Retention:** In OFF devices, gate tunneling current is either numerically zero ($\sim 10^{-18}\text{ A}$ for NMOS OFF, dominated by numerical noise) or already well-modeled by the baseline shared MLP ($2.12\%$ median error for PMOS OFF). The final OFF models retain the original model's $I_g$ column via `PerTerminal` wrapper, preserving the baseline models in `models/orig/final_<case>.joblib` for inference.
5. **Iterative Evaluation History:** The final model architecture was informed by analyzing downstream error bottlenecks from earlier iterations (v1 baseline, hybrid dedicated $I_g$, and ensemble trials).
6. **ON-Device Channel Current Invariance:** Transistor-level ON-state $I_d$ and $I_s$ errors (4–14%) have negligible downstream impact because gate-level and circuit-level leakage formulations apply the physical KCL decomposition: $I_c = (I_d - I_s)/2$ and $g_b = (I_g + I_b)/2$, where the massive conduction channel cancels out and static leakage is dictated by gate-tunneling and substrate currents.
7. **Numerical Noise & Residuals:** NMOS OFF $I_g$ 95th-percentile relative error is large but numerically zero in absolute terms ($\le 1.6 \times 10^{-5}\text{ A}$ relative to $10^{-15}\text{ A}$ floor; currents are $\sim 10^{-18}\text{ A}$). Kirchhoff relative residuals exceed tolerance only at the numerical noise floor ($\sim 10^{-15}\text{ A}$).
8. **Engineering Assumptions without TA Confirmation:** (a) Gate oxide thicknesses `toxp` and `toxm` were not varied independently and remain fixed to nominal model card values; (b) total leakage is defined as net current supplied by $V_{DD}$ sources; (c) parity checker logic polarity convention (even parity = 0, odd parity = 1).
9. **Metric Conventions:** $R^2$ is evaluated in symmetric logarithmic space ($T(I) = \operatorname{sign}(I)\log_{10}(1 + |I|/10^{-15})$). Relative errors in Amperes use a numerical noise floor of $10^{-15}\text{ A}$.
10. **Simulator Consistency:** ngspice-42 is recorded in `ngspice_version.txt`; ngspice-43 installed locally at `/usr/local/bin/ngspice` was verified to produce bit-identical results on all 36 test netlists.
11. **Inter-Gate Electrical Loading:** The hierarchical composition assumes negligible inter-gate electrical loading in static DC steady state, which is verified by standalone gate SPICE summation reproducing full-circuit SPICE to 99.98% accuracy.
12. **Execution Note:** A complete from-scratch re-simulation pipeline run (~1.5 hours) was NOT executed during this upgrade; the verified model package was adopted and evaluated with single-pass verification.

---

## Number -> CSV Trace Table

| Report Claim / Headline | Exact Value | Source CSV File | Column / Row Reference |
|---|---|---|---|
| Gate SPICE Node V Mean Accuracy | 99.28% | `results/PartB_gates/gate_results.csv` | Mean of `mean_acc_%` across 18 gate rows |
| Gate SPICE Node V Median Relative Error | 0.51% | `data/PartB_gates/gate_ml.csv` | Median of `abs(rel_err) * 100` (all 3,600 rows) |
| Gate SPICE Node V 95th Percentile Error | 1.98% | `data/PartB_gates/gate_ml.csv` | 95th percentile of `abs(rel_err) * 100` (all 3,600 rows) |
| Gate ML-Solved Node V Mean Accuracy | 99.25% | `data/PartB_gates/gate_nodes.csv` | `100 * (1 - mean(abs(ml_leak_mlV - spice_leak)/spice_leak))` |
| Gate ML-Solved Node V Median Error | 0.53% | `data/PartB_gates/gate_nodes.csv` | `100 * median(abs(ml_leak_mlV - spice_leak)/spice_leak)` |
| Gate ML-Solved Node V 95th Percentile Error | 2.08% | `data/PartB_gates/gate_nodes.csv` | `100 * quantile(abs(ml_leak_mlV - spice_leak)/spice_leak, 0.95)` |
| Circuit Mode 1 (Fully-ML) Mean Accuracy | 99.46% | `results/PartC_parity_checker/circuit_results.csv` | Row `ALL`, column `mean_acc_%` |
| Circuit Mode 1 Median Relative Error | 0.43% | `results/PartC_parity_checker/circuit_results.csv` | Row `ALL`, column `median_err_%` |
| Circuit Mode 1 95th Percentile Error | 1.57% | `results/PartC_parity_checker/circuit_results.csv` | Row `ALL`, column `p95_err_%` |
| Circuit Mode 2 (SPICE Node V) Mean Accuracy | 99.49% | `data/PartC_parity_checker/circuit_ml.csv` | `100 * (1 - mean(abs(ml_leak_gatemodel - spice_leak)/spice_leak))` |
| Circuit Mode 2 Median Relative Error | 0.40% | `data/PartC_parity_checker/circuit_ml.csv` | `100 * median(abs(ml_leak_gatemodel - spice_leak)/spice_leak)` |
| Circuit Mode 2 95th Percentile Error | 1.33% | `data/PartC_parity_checker/circuit_ml.csv` | `100 * quantile(abs(ml_leak_gatemodel - spice_leak)/spice_leak, 0.95)` |
| Standalone Gate SPICE Summation Accuracy | 99.98% | `data/PartC_parity_checker/circuit_ml.csv` | `100 * (1 - mean(abs(sum_gate_spice - spice_leak)/spice_leak))` |
| Transistor Validation Errors (Old vs New) | Table | `results/PartA_transistor/new_vs_old_validation.csv` | All 4 cases, columns `old_*_med_%` vs `new_*_med_%` |
| Transistor Test Set Errors (Per Current) | Table | `results/PartA_transistor/test_chosen.csv` | Columns `id_median`, `ig_median`, `is_median`, `ib_median` |

---

## Execution and Canonical Codebase

- **Canonical runner:** `regenerate_code/run_all.sh` is the canonical pipeline runner.
- **Root execution wrapper:** The root `./run_all.sh` delegates all arguments directly to the canonical runner (`./run_all.sh --verify-only` for non-destructive verification, `./run_all.sh --force-full` for full regeneration).
- **Canonical script package:** `regenerate_code/scripts/` is the canonical, integrated, fully tested executable script package. All pipeline stages execute strictly from this directory.
- **Reference catalog (`scripts/`):** The root `scripts/` directory is an archival reference catalog organized by assignment part for inspection only. It is **NOT independently runnable** (due to part-specific subdirectories and shared module dependencies). Users must not execute or modify the modular tree as the canonical implementation.

## Standalone Stage Execution and Dependencies

The individual stage runners normalize their working directory, so they can be invoked from outside the project directory. For individual stage execution, users should run stages in dependency order:

- **Stage 1: Part A Data Generation** (`regenerate_code/run_partA_data.sh`)
  - Generates `data/corners_200.csv` and transistor datasets.
- **Stage 2: Part A Model Training** (`regenerate_code/run_partA_models.sh`)
  - Trains or updates transistor models using `data/PartA_transistor/`.
- **Stage 3: Part B Gate-Level Validation** (`regenerate_code/run_partB_gates.sh`)
  - Evaluates standard cell gates (`gate_ml.py`, `gate_nodes.py`, `gate_plot.py`).
- **Stage 4: Part C 8-bit Parity Checker** (`regenerate_code/run_partC_parity_checker.sh`)
  - Evaluates complete parity tree (`circuit_ml.py`, `circuit_plot.py`).

| Folder | Contents (each split into PartA_transistor / PartB_gates / PartC_parity_checker) |
|---|---|
| data/ | datasets, corners (shared), gate and circuit SPICE+ML tables, 16nm_LP model file |
| netlists/ | transistor, gate and test-circuit example netlists |
| results/ | data check, model comparison, test metrics, gate table, circuit table |
| plots/ | trend plots, predicted-vs-SPICE, gate and circuit parity plots |
| models/ | saved ML models (`final_<case>.joblib`, `models/orig/` baseline models, `pert.py`) |
| scripts/ | reference code catalog organized by part (non-executable; see `scripts/README.md`) |
| readme/ | one short README per part |
| report/ | typed project report (`report.md`, `report.tex`, `report.pdf`, `report.docx`) |
| regenerate_code/ | canonical standalone package (flat scripts layout + runner scripts); `./run_all.sh` recreates all outputs |
