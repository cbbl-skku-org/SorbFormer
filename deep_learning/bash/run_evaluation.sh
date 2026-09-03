#!/bin/bash
# run_evaluation.sh
# Usage: bash run_evaluation.sh [results_folder]

RESULTS_DIR=${1:-"final_models_rcf"}

echo "Running Model Evaluation on $RESULTS_DIR"

python scripts/03_evaluate_performance.py --results_folder $RESULTS_DIR
