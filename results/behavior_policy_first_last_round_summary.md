## Behavior policy: first vs last round

Values below come from the existing `adaptive_behavior_by_round.csv` artifacts.
They are cumulative context-weighted evaluations of the fitted behavior policy at the first and last available round bucket.

### attempts

- first round bucket: `0` (round 1)
- last round bucket: `29` (round 30)

| estimator | first | last | delta (last - first) |
| --- | ---: | ---: | ---: |
| IPS | 0.786 | 1.782 | +0.996 |
| SNIPS | 0.894 | 1.660 | +0.766 |
| DR | 0.908 | 1.640 | +0.733 |
| MIS | 0.786 | 1.661 | +0.876 |
| DM | 0.907 | 1.521 | +0.613 |

### skill_builder

- first round bucket: `0` (round 1)
- last round bucket: `49` (round 50)

| estimator | first | last | delta (last - first) |
| --- | ---: | ---: | ---: |
| IPS | 0.573 | 0.732 | +0.159 |
| SNIPS | 0.572 | 0.730 | +0.158 |
| DR | 0.585 | 0.742 | +0.157 |
| MIS | 0.573 | 0.733 | +0.160 |
| DM | 0.704 | 0.747 | +0.043 |

### assistments8000

- first round bucket: `0` (round 1)
- last round bucket: `49` (round 50)

| estimator | first | last | delta (last - first) |
| --- | ---: | ---: | ---: |
| IPS | 0.574 | 0.653 | +0.080 |
| SNIPS | 0.571 | 0.644 | +0.073 |
| DR | 0.571 | 0.644 | +0.073 |
| MIS | 0.574 | 0.655 | +0.081 |
| DM | 0.652 | 0.666 | +0.015 |

### pix

- first round bucket: `0` (bucket 1)
- last round bucket: `19` (bucket 20)
- note: this experiment truncates to the first 30 interactions, then caps them into 20 round buckets; the final value is therefore for the last bucket, not the raw 30th round

| estimator | first | last | delta (last - first) |
| --- | ---: | ---: | ---: |
| IPS | 3.361 | 1.968 | -1.393 |
| SNIPS | 1.735 | 2.285 | +0.550 |
| DR | 1.596 | 2.246 | +0.651 |
| MIS | 3.361 | 2.250 | -1.112 |
| DM | 1.398 | 2.197 | +0.799 |

### Short interpretation

- `attempts` shows the largest improvement in the behavior policy over rounds.
- `skill_builder` and `assistments8000` improve more moderately.
- `pix` is different: raw `IPS` and `MIS` are very high at the first bucket and decrease by the last bucket, while `SNIPS`, `DR`, and `DM` increase strongly. This indicates that the early PIX behavior values are driven by unstable raw importance weighting, whereas the normalized and model-based estimators suggest improvement over time.
