## Rasch calibration on held-out user splits

The calibration diagnostics were computed from the held-out user splits using the cached KTM `proficiency` and `difficulties` values. For each interaction, the Rasch probability was

`p(correct | theta, delta) = sigmoid(theta - delta)`

and we reported:
- `logloss`
- `brier`
- `auc`
- `cal_gap_q10`: mean absolute calibration gap across 10 equal-frequency bins
- `ece_q10`: weighted expected calibration error across the same bins

### Held-out test metrics

| dataset | n_test_rows | logloss | brier | auc | cal_gap_q10 | ece_q10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attempts | 26,033 | 0.250923 | 0.071355 | 0.819724 | 0.021209 | 0.021203 |
| skill_builder | 47,246 | 0.527817 | 0.175654 | 0.712548 | 0.032447 | 0.032441 |
| assistments8000 | 9,651 | 0.567377 | 0.190683 | 0.787102 | 0.111218 | 0.111224 |
| pix | 298,031 | 0.515682 | 0.174271 | 0.812710 | 0.028048 | 0.028047 |

### Interpretation

`assistments8000` is the clear calibration outlier. Its AUC is still reasonably strong, so the Rasch score is not useless for ranking, but its calibration gap is much larger than on the other datasets. This is the main reason to treat the direct-method values more cautiously on `assistments8000`: the DM estimator uses the Rasch reward model directly, so poor calibration on the probability scale makes the resulting DM estimates less reliable.

### Code paths that matter

The direct-method estimator uses the Rasch reward model directly:
- `/Users/samuelgirard/work/geometric_flow/offpolicy_ktm_pipeline/run_last_round_simple.py`
  - `q_logged = sigmoid(theta - delta) * (delta - delta_min)` at lines 236--244
  - `dm = sum(c_i * q_pi) / sum(c_i)` at line 249
- `/Users/samuelgirard/work/geometric_flow/offpolicy_ktm_pipeline/src/estimators_and_training.py`
  - `dm_policy_value(...)` at lines 516--541

The reusable calibration script is:
- `/Users/samuelgirard/work/geometric_flow/eval/rasch_calibration_user_split.py`
