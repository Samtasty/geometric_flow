# Off-Policy Evaluation Experiment Report — v2

**Date:** 2026-03-12
**Status:** All four datasets complete

---

## 1. Objective

Learn a better item-assignment policy from logged student interaction data using off-policy evaluation (OPE). Given a student's current proficiency θ and the round they are on, we want to find a policy π(δ | θ) over item difficulty δ that maximises expected reward — where reward = `correct × (δ − δ_min)`, encouraging assignment of harder items that students can answer correctly.

---

## 2. Data and Preprocessing

### Datasets

| Dataset | Source | Raw rows | Truncation | Students (total) |
|---------|--------|----------|------------|-----------------|
| **attempts** | `attempts_sensitivity_0p10/ktm_dataframe.csv` | ~234k | round ≤ 30 | ~20,556 |
| **assistments8000** | `assistments8000_trunc50/ktm_dataframe.csv` | ~98k | round ≤ 50 | ~4,066 |
| **skill_builder** | `skill_builder_trunc50/ktm_dataframe.csv` | ~473k | round ≤ 50 | ~14,567 |
| **pix** | `pix_mapping/.../ktm_dataframe.csv` | ~3.2M | last 30 rounds/student (tail cap) | 100,000 (80k/10k/10k) |

Each dataset starts as a KTM (Knowledge Tracing Model) dataframe with columns:
- `user`, `item`, `correct` — student, question, outcome (0/1)
- `proficiency` (θ) — IRT-estimated student ability
- `difficulties` (δ) — IRT-estimated item difficulty
- `order_sequence` — the attempt number for this student (0-indexed round)
- `answer_number`, `sequence_position`, `sequence_length`

### IRT Model

Item Response Theory predicts correctness as:
```
P(correct | θ, δ) = sigmoid(θ − δ)
```
Proficiency θ and difficulty δ are estimated by the KTM pipeline prior to these experiments.

### User-Level Train / Validation / Test Split

Students are split **80% / 10% / 10%** at the user level (not row level), ensuring no data leakage across splits. All rows for a given student land in exactly one split.

- **Train**: behavior policy fitting + policy learning
- **Val**: held out (not used in these experiments)
- **Test**: evaluation only — estimators computed on test split

For **pix**, the dataset has ~3.2M rows across ~100k students (split 80k/10k/10k). Each student's history is capped to their **last 30 rounds** (`round_cap_strategy=tail`), so students with more than 30 attempts retain only the most recent 30. The maximum `order_sequence` in the dataset is 49 (students who have done 50+ rounds retain rounds 20–49).

---

## 3. Behavior Policy Fitting

For each dataset and each round `k`, we fit a truncated Gaussian behavior policy:
```
π_b(δ | θ, k) = TruncNormal(μ_k(θ), σ_k(θ)²; [δ_min − ε, δ_max + ε])
```
where:
- `μ_k(θ) = β_μ,0 + β_μ,1 · θ` (degree-1 polynomial)
- `log σ_k(θ) = β_σ,0 + β_σ,1 · θ + β_σ,2 · θ²` (degree-2 polynomial)
- Support bounds are `δ_min − 1e-4`, `δ_max + 1e-4` (expanded slightly to avoid float32 boundary artifacts)

Fitting is done via **L-BFGS-B** minimising the truncated Gaussian negative log-likelihood, fitted **on the train split only**.

### Auto-Exclusion of Poorly-Fitted Rounds

Rounds with **RMSE > 5.0** are automatically excluded from training data and MIS estimation. This catches degenerate optimizer solutions (e.g., μ ≪ δ_min with huge σ → near-uniform distribution). Affected rounds:

| Dataset | Excluded rounds | RMSE of excluded |
|---------|----------------|-----------------|
| attempts | 5, 6, 7 | 16.1, 30.2, 30.2 |
| assistments8000 | none | — |
| skill_builder | none | — |
| pix | 0 | 16.4 |

### Behavior Policy Fit Quality (first vs. last round, train split)

| Dataset | First round | Last round | First RMSE | Last RMSE |
|---------|------------|-----------|------------|-----------|
| attempts | 0 | 29 | 0.637 | 1.016 |
| assistments8000 | 0 | 49 | 0.309 | 0.352 |
| skill_builder | 0 | 49 | 0.531 | 0.552 |
| pix | 0* | 49 | 16.4* | 0.686 |

\* Round 0 excluded (degenerate fit); first valid round is 1.

RMSE increases slightly with round number due to smaller sample sizes (student attrition), but remains well-controlled for all non-excluded rounds.

---

## 4. Off-Policy Reward and Propensity

### Reward
```
r_i = correct_i × (δ_i − δ_min)
```
`δ_min` is the global minimum difficulty across the full dataset. This rewards assigning harder items that students answer correctly.

### Propensity

The behavior propensity for each logged interaction is:
```
log π_b(δ_i | θ_i, k_i) = log TruncNormal(δ_i; μ_{k_i}(θ_i), σ_{k_i}(θ_i))
```
Computed using `log_ndtr` (scipy) with the survival-function branch for numerical stability in far tails:
- For α > 0: use `log Z = log[Φ(-α) − Φ(-β)]` (SF branch)
- For α ≤ 0: use `log Z = log[Φ(β) − Φ(α)]` (CDF branch)

This matches exactly the implementation in `GlobalGaussianPolicy.log_prob` (PyTorch, float64 precision).

---

## 5. Context Reweighting

We evaluate policies with respect to the **last round's θ distribution** as the target context. A histogram density ratio `c_i = q_target(θ_i) / q_logged(θ_i)` is computed for each sample, where:
- `q_target` = empirical θ histogram of the target round (chosen from the first half of rounds by minimum Jensen-Shannon divergence to the last round)
- `q_logged` = empirical θ histogram of the round each sample came from

All IPS/SNIPS/DR/MIS estimates are implicitly weighted by `c_i`.

### Target Round Selection

| Dataset | Last round | Target round (first half) | Mean JS divergence |
|---------|-----------|--------------------------|-------------------|
| attempts | 29 | 15 | 0.0499 |
| assistments8000 | 49 | 24 | 0.0138 |
| skill_builder | 49 | 24 | 0.0044 |
| pix | 49 | 23 | — (not stored) |

---

## 6. Learned Policy

A single **GlobalGaussianPolicy** is learned (not per-round), parameterized by:
```
π_new(δ | θ) = TruncNormal(μ(θ), σ(θ)²; [δ_min, δ_max])
```
with the same polynomial parameterization as the behavior policy (degree 1 for μ, degree 2 for σ).

### Training

| Hyperparameter | Value |
|---------------|-------|
| Epochs | 20 |
| Batch size | 65,536 (pix: 65,536; others: 16,384–65,536) |
| Learning rate | 0.005 |
| Max importance weight | 20.0 (clipping) |
| Optimizer | Adam |

Objectives trained: **IPS**, **SNIPS**, **DR**, **MIS**
Training is done on the train split; user/val/test splits are fixed (reused from prior runs where applicable).

---

## 7. Estimators

All estimators evaluate the policy value `V(π) = E[r · w]` where `w = π_new(δ|θ) / π_b(δ|θ)`:

| Estimator | Formula |
|-----------|---------|
| **IPS** | `Σ w_i r_i / n` |
| **SNIPS** | `Σ w_i r_i / Σ w_i` (self-normalized) |
| **DR** | IPS + DM residual correction |
| **MIS** | Marginal IS — denominator is mixture over all rounds' behavior policies |
| **DM** | Direct method — IRT-based reward predictor only (no IS) |
| **ESS_logged** | Effective sample size `(Σ w_i)² / Σ w_i²` |
| **ESS_mix** | ESS with MIS weights |

All estimates use the **cumulative** train/eval scope (all rounds' data, not just last round).

---

## 8. Results (Test Split)

### Behavior Policy Value — First and Last Round

The behavior policy IPS estimate (cumulative, over all rounds in test) is used as the baseline.

| Dataset | Behavior IPS | Behavior SNIPS | Behavior DR | Behavior MIS |
|---------|-------------|---------------|------------|-------------|
| attempts | 2.012 | 1.989 | 2.012 | 2.103 |
| assistments8000 | 0.657 | 0.635 | 0.631 | 0.650 |
| skill_builder | 0.882 | 0.883 | 0.877 | 0.884 |
| pix | 1.782 | 2.181 | 2.142 | 2.163 |

> Note: these are **cumulative** estimates over all rounds (context-reweighted toward the last round's θ distribution). The "last-round-only" value can be read as the DM estimate (direct method baseline), since only round-29/49 samples contribute.

### Learned Policy Results — attempts (trunc@30, excl. rounds 5/6/7)

| Policy | IPS | SNIPS | DR | MIS | DM |
|--------|-----|-------|----|-----|----|
| behavior | 2.012 | 1.989 | 2.012 | 2.103 | 2.013 |
| learned_ips | **2.355** | 2.371 | 2.371 | 2.677 | 2.235 |
| learned_snips | 2.256 | 2.440 | 2.451 | 2.798 | 2.321 |
| learned_dr | 2.220 | **2.442** | **2.456** | 2.785 | 2.334 |
| learned_mis | 2.348 | 2.463 | 2.472 | **2.949** | 2.321 |
| optimal_irt | 1.693 | 2.353 | 2.399 | 2.456 | **2.442** |

Improvement over behavior (best SNIPS): **+23%**

### Learned Policy Results — assistments8000 (trunc@50)

| Policy | IPS | SNIPS | DR | MIS | DM |
|--------|-----|-------|----|-----|----|
| behavior | 0.657 | 0.635 | 0.631 | 0.650 | 0.643 |
| learned_ips | **0.727** | 0.685 | 0.695 | 0.724 | 0.611 |
| learned_snips | 0.704 | **0.685** | 0.694 | 0.704 | 0.638 |
| learned_dr | 0.698 | 0.684 | **0.694** | 0.698 | **0.648** |
| learned_mis | 0.707 | 0.622 | 0.596 | **0.716** | 0.697 |
| optimal_irt | 0.587 | 0.611 | 0.630 | 0.580 | 0.770 |

Improvement over behavior (best SNIPS): **+8%**

### Learned Policy Results — skill_builder (trunc@50)

| Policy | IPS | SNIPS | DR | MIS | DM |
|--------|-----|-------|----|-----|----|
| behavior | 0.882 | 0.883 | 0.877 | 0.884 | 0.884 |
| learned_ips | **1.299** | 1.081 | 1.077 | **1.298** | 1.086 |
| learned_snips | 1.191 | **1.093** | **1.098** | 1.194 | 1.117 |
| learned_dr | 1.040 | 1.086 | 1.104 | 1.047 | **1.146** |
| learned_mis | 1.298 | 1.081 | 1.077 | 1.298 | 1.088 |
| optimal_irt | 0.759 | 1.054 | 1.094 | 0.776 | 1.172 |

Improvement over behavior (best SNIPS): **+24%**

> Note: The very high IPS for `learned_ips` on skill_builder (1.299 vs 0.882) relative to SNIPS (1.081) indicates high-variance IS weights. SNIPS and DR are more reliable estimates here.

### Learned Policy Results — pix (last 30 rounds/student, 100k users, excl. round 0)

| Policy | IPS | SNIPS | DR | MIS | DM |
|--------|-----|-------|----|-----|----|
| behavior | 1.782 | 2.181 | 2.142 | 2.163 | 2.155 |
| learned_ips | **2.812** | 3.007 | 2.996 | 3.109 | 2.663 |
| learned_snips | 2.696 | **3.046** | **3.017** | 2.984 | 2.692 |
| learned_dr | 2.745 | 3.045 | 3.024 | 3.027 | **2.697** |
| learned_mis | 2.737 | 2.599 | 2.603 | **3.376** | 2.485 |
| optimal_irt | 2.216 | 2.889 | 2.907 | 2.634 | 2.736 |

Improvement over behavior (best SNIPS): **+40%**

> Round 0 was auto-excluded (RMSE = 16.4 > threshold 5.0). The large gap between behavior IPS (1.782) and SNIPS (2.181) reflects high variance in the logged distribution — SNIPS/DR are more reliable here.

---

## 9. Learned Policy Coefficients

### μ(θ) = β_μ,0 + β_μ,1 · θ

| Dataset | Objective | β_μ,0 | β_μ,1 |
|---------|-----------|-------|-------|
| attempts | ips | -1.133 | 1.430 |
| attempts | snips | -1.141 | 1.647 |
| attempts | dr | -1.115 | 1.646 |
| attempts | mis | -1.111 | 1.532 |
| assistments8000 | ips | -0.525 | 0.215 |
| assistments8000 | snips | -0.480 | 0.480 |
| assistments8000 | dr | -0.413 | 0.419 |
| assistments8000 | mis | 0.061 | 0.318 |
| skill_builder | ips | 0.317 | 0.653 |
| skill_builder | snips | 0.378 | 1.011 |
| skill_builder | dr | 0.569 | 1.231 |
| skill_builder | mis | 0.332 | 0.641 |
| pix | ips | -0.169 | 1.080 |
| pix | snips | -0.226 | 0.974 |
| pix | dr | -0.202 | 0.915 |
| pix | mis | 0.170 | 0.107 |

All datasets learn a **positive β_μ,1** (assign harder items to higher-proficiency students), consistent with IRT theory. The intercept β_μ,0 varies with the dataset's difficulty range.

---

## 10. Key Technical Notes

### Numerical Stability Fixes (v2)

1. **Truncated Gaussian support expansion**: `δ_min − 1e-4`, `δ_max + 1e-4` to prevent float32 boundary rejection (observations at exactly `δ_min` in float64 become slightly outside support after float32 round-trip).

2. **scipy / PyTorch propensity alignment**: `propensity.py` now uses `log_ndtr` with identical survival-function branch switching as `GlobalGaussianPolicy.log_prob`, ensuring behavior policy IS weights average to ≈ 1.0.

3. **Degenerate behavior policy detection**: Rounds where the optimizer finds μ ≪ δ_min with huge σ (near-uniform approximation) are flagged via RMSE > 5.0 and excluded from data and MIS mixture.

---

## 11. Output Files (per run)

| File | Contents |
|------|----------|
| `estimator_results_by_split.csv` | IPS/SNIPS/DR/MIS/DM per policy and split |
| `gaussian_fit_train_by_order.csv` | Behavior policy coefficients and fit stats per round |
| `learned_policy_coefficients.csv` | β_μ, β_σ for each trained objective |
| `training_history.csv` | Train loss per epoch per objective |
| `split_summary.csv` | n_users, n_rows, n_eval per split |
| `run_summary.csv` | All hyperparameters and metadata |
| `user_split_assignments.csv` | User→split mapping |
| `round_bucket_mapping.csv` | Round mapping and obs counts |
