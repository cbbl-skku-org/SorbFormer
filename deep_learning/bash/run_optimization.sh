#!/bin/bash
# run_optimization.sh
# Usage: bash run_optimization.sh [target] [metric] [model_extractor]
# Example: bash run_optimization.sh rcf r2 biot5-plus-base

TARGET=${1:-rcf}
METRIC=${2:-r2}
MODEL=${3:-biot5-plus-base}

echo "Starting Hyperparameter Optimization..."
echo "Target: $TARGET"
echo "Metric: $METRIC"
echo "Model Extractor: $MODEL"

# Determine datasets
if [ "$TARGET" = "rcf" ]; then
    TRAIN_CSV="data/rcf_train_val.csv"
    TEST_CSV="data/rcf_test.csv"
    METADATA="metadata_rcf.json"
else
    TRAIN_CSV="data/tscf_train_val.csv"
    TEST_CSV="data/tscf_test.csv"
    METADATA="metadata_tscf.json"
fi

python scripts/01_optimize_models.py \
    --train_csv $TRAIN_CSV \
    --test_csv $TEST_CSV \
    --target $TARGET \
    --metadata_file $METADATA \
    --mol_extractor_type $MODEL \
    --study_name "SorbFormer_${TARGET}_${METRIC}" \
    --n_trials 100 \
    --save_folder "results_${TARGET}_${METRIC}"
