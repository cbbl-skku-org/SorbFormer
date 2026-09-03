#!/bin/bash
# run_training.sh
# Usage: bash run_training.sh [target] [model_extractor] [fusion_type]
# Example: bash run_training.sh rcf biot5-plus-base cross_attention

TARGET=${1:-rcf}
MODEL=${2:-biot5-plus-base}
FUSION=${3:-cross_attention}

echo "Starting Model Training (Refit)..."
echo "Target: $TARGET"
echo "Model Extractor: $MODEL"
echo "Fusion Type: $FUSION"

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

# The training script assumes we already have a top_k_trials config from optimization.
# For public release, users will likely train the exact best config.
python scripts/02_train_best_models.py \
    --train_csv $TRAIN_CSV \
    --test_csv $TEST_CSV \
    --target $TARGET \
    --metadata_file $METADATA \
    --mol_extractor_type $MODEL \
    --fusion_type $FUSION \
    --study_name "SorbFormer_${TARGET}_best" \
    --save_folder "final_models_${TARGET}" \
    --n_trials 1
