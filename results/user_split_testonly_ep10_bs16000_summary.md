## Held-out user-split comparison

These numbers use the refreshed behavior-policy evaluation after fixing the logger reconstruction bug.

Configuration shared by all runs:
- user split `80/10/10`
- evaluation on `test` only
- objectives `ips`, `snips`, `dr`
- `10` epochs
- `batch_size=16000`
- `pix` also includes a reused `mis` run

### Behavior policy: first vs last round

These values are computed on the train split, using the first and last round-specific behavior models under the same fixed target context.

| dataset | first round | last round | IPS_1 | IPS_T | dIPS | SNIPS_1 | SNIPS_T | dSNIPS | DR_1 | DR_T | dDR | MIS_1 | MIS_T | dMIS | DM_1 | DM_T | dDM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attempts | 0 | 29 | 1.690 | 2.333 | 0.643 | 1.693 | 2.155 | 0.462 | 1.723 | 2.130 | 0.408 | 1.690 | 2.256 | 0.566 | 1.645 | 2.008 | 0.363 |
| skill_builder | 0 | 49 | 0.661 | 0.876 | 0.214 | 0.661 | 0.876 | 0.215 | 0.657 | 0.872 | 0.215 | 0.661 | 0.878 | 0.217 | 0.823 | 0.882 | 0.059 |
| assistments8000 | 0 | 49 | 0.575 | 0.650 | 0.075 | 0.575 | 0.643 | 0.068 | 0.576 | 0.641 | 0.065 | 0.575 | 0.651 | 0.075 | 0.616 | 0.648 | 0.031 |
| pix | 0 | 29 | 0.324 | 1.745 | 1.420 | 0.324 | 2.189 | 1.865 | 0.374 | 2.147 | 1.773 | 0.324 | 2.143 | 1.819 | 0.362 | 2.155 | 1.793 |

### Final test results

| dataset | policy | IPS | SNIPS | DR | MIS | DM | ESS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attempts | behavior | 2.271 | 2.139 | 2.114 | -- | 2.011 | 4541 |
| attempts | optimal_irt_gaussian | 1.791 | 2.444 | 2.450 | -- | 2.437 | 1574 |
| attempts | learned_ips | 2.915 | 2.517 | 2.535 | -- | 2.251 | 3653 |
| attempts | learned_snips | 2.866 | 2.536 | 2.561 | -- | 2.286 | 3510 |
| attempts | learned_dr | 2.801 | 2.544 | 2.570 | -- | 2.307 | 3363 |
| skill_builder | behavior | 0.882 | 0.883 | 0.877 | -- | 0.884 | 43126 |
| skill_builder | optimal_irt_gaussian | 0.759 | 1.053 | 1.094 | -- | 1.171 | 3950 |
| skill_builder | learned_ips | 1.294 | 1.079 | 1.076 | -- | 1.082 | 18519 |
| skill_builder | learned_snips | 1.248 | 1.089 | 1.089 | -- | 1.098 | 17761 |
| skill_builder | learned_dr | 1.187 | 1.091 | 1.098 | -- | 1.121 | 15354 |
| assistments8000 | behavior | 0.657 | 0.638 | 0.634 | -- | 0.643 | 6172 |
| assistments8000 | optimal_irt_gaussian | 0.587 | 0.611 | 0.630 | -- | 0.770 | 538 |
| assistments8000 | learned_ips | 0.681 | 0.671 | 0.679 | -- | 0.644 | 7070 |
| assistments8000 | learned_snips | 0.679 | 0.672 | 0.681 | -- | 0.649 | 6885 |
| assistments8000 | learned_dr | 0.675 | 0.671 | 0.679 | -- | 0.655 | 6884 |
| pix | behavior | 1.731 | 2.184 | 2.144 | 2.143 | 2.155 | 136862 |
| pix | optimal_irt_gaussian | 2.148 | 2.889 | 2.902 | 2.624 | 2.736 | 27549 |
| pix | learned_ips | 2.735 | 2.984 | 2.975 | 3.092 | 2.653 | 33993 |
| pix | learned_snips | 2.610 | 3.045 | 3.007 | 2.967 | 2.690 | 26113 |
| pix | learned_dr | 2.673 | 3.043 | 3.014 | 3.027 | 2.696 | 29612 |
| pix | learned_mis | 2.647 | 2.601 | 2.600 | 3.346 | 2.486 | 62803 |

### Short comparison

- `attempts`: all learned policies remain above behavior; `learned_dr` is best on `SNIPS`, `DR`, and `DM`, while `learned_ips` is best only on `IPS`.
- `skill_builder`: same pattern, with smaller gains; `learned_dr` is the most stable policy overall.
- `assistments8000`: improvements over behavior remain small; `learned_snips` is best on `SNIPS/DR`, but the oracle still dominates on `DM`.
- `pix`: `learned_ips` is best on `IPS`, `learned_snips` on `SNIPS`, `learned_dr` on `DR/DM`, and `learned_mis` on `MIS`.

### Improvement over behavior

All numbers below are policy value minus the behavior policy value on the same test split.

| dataset | policy | dIPS | dSNIPS | dDR | dMIS | dDM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| attempts | optimal_irt_gaussian | -0.480 | 0.305 | 0.336 | -- | 0.426 |
| attempts | learned_ips | 0.644 | 0.378 | 0.421 | -- | 0.241 |
| attempts | learned_snips | 0.595 | 0.396 | 0.447 | -- | 0.275 |
| attempts | learned_dr | 0.530 | 0.404 | 0.456 | -- | 0.296 |
| skill_builder | optimal_irt_gaussian | -0.122 | 0.171 | 0.217 | -- | 0.287 |
| skill_builder | learned_ips | 0.413 | 0.197 | 0.200 | -- | 0.198 |
| skill_builder | learned_snips | 0.366 | 0.206 | 0.212 | -- | 0.214 |
| skill_builder | learned_dr | 0.306 | 0.208 | 0.221 | -- | 0.236 |
| assistments8000 | optimal_irt_gaussian | -0.070 | -0.028 | -0.004 | -- | 0.128 |
| assistments8000 | learned_ips | 0.024 | 0.033 | 0.045 | -- | 0.001 |
| assistments8000 | learned_snips | 0.022 | 0.034 | 0.047 | -- | 0.006 |
| assistments8000 | learned_dr | 0.018 | 0.033 | 0.046 | -- | 0.012 |
| pix | optimal_irt_gaussian | 0.418 | 0.705 | 0.758 | 0.481 | 0.580 |
| pix | learned_ips | 1.004 | 0.799 | 0.831 | 0.949 | 0.498 |
| pix | learned_snips | 0.879 | 0.861 | 0.863 | 0.823 | 0.535 |
| pix | learned_dr | 0.942 | 0.859 | 0.870 | 0.884 | 0.540 |
| pix | learned_mis | 0.916 | 0.416 | 0.456 | 1.203 | 0.330 |

### Context-target matching

| dataset | last round | target round | JS(target,last) |
| --- | ---: | ---: | ---: |
| attempts | 29 | 14 | 0.0808 |
| skill_builder | 49 | 24 | 0.00436 |
| assistments8000 | 49 | 24 | 0.01384 |
| pix | 29 | 14 | 0.01018 |
