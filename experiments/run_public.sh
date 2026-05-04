#!/usr/bin/env bash
# Run OPE experiments for the three public datasets.
# Usage: bash experiments/run_public.sh
#
# Expected input CSVs (ktm_dataframe.csv) at:
#   results/attempts_sensitivity_0p10/ktm_dataframe.csv
#   results/assistments8000_trunc50/ktm_dataframe.csv
#   results/skill_builder_trunc50/ktm_dataframe.csv
#
# These are produced by the KTM pre-processing pipeline.
# Pre-processed CSVs are available in the supplementary data archive.
set -e
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python3"

echo "========================================"
echo " attempts  (truncate @ 30 rounds)"
echo "========================================"
$PYTHON -u experiments/run_ope.py \
  --input-csv results/attempts_sensitivity_0p10/ktm_dataframe.csv \
  --input-is-ktm \
  --dataset-name attempts \
  --output-dir experiments/results/attempts \
  --truncate-at-round 30 \
  --max-rounds 50 \
  --rmse-threshold 5.0 \
  --objectives ips snips dr mis \
  --epochs 20 --batch-size 32768 --lr 0.005 --max-weight 20.0

echo ""
echo "========================================"
echo " assistments8000  (truncate @ 50 rounds)"
echo "========================================"
$PYTHON -u experiments/run_ope.py \
  --input-csv results/assistments8000_trunc50/ktm_dataframe.csv \
  --input-is-ktm \
  --dataset-name assistments8000 \
  --output-dir experiments/results/assistments8000 \
  --truncate-at-round 50 \
  --max-rounds 50 \
  --rmse-threshold 5.0 \
  --objectives ips snips dr mis \
  --epochs 20 --batch-size 32768 --lr 0.005 --max-weight 20.0

echo ""
echo "========================================"
echo " skill_builder  (truncate @ 50 rounds)"
echo "========================================"
$PYTHON -u experiments/run_ope.py \
  --input-csv results/skill_builder_trunc50/ktm_dataframe.csv \
  --input-is-ktm \
  --dataset-name skill_builder \
  --output-dir experiments/results/skill_builder \
  --truncate-at-round 50 \
  --max-rounds 50 \
  --rmse-threshold 5.0 \
  --objectives ips snips dr mis \
  --epochs 20 --batch-size 32768 --lr 0.005 --max-weight 20.0

echo ""
echo "All done. Reports in experiments/results/*/report.html"
echo ""
echo "To plot learned policies:"
echo "  python experiments/plot_policies.py --results-dir experiments/results/attempts"
echo "  python experiments/plot_policies.py --results-dir experiments/results/assistments8000"
echo "  python experiments/plot_policies.py --results-dir experiments/results/skill_builder"
