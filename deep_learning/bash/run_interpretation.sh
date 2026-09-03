#!/bin/bash
# run_interpretation.sh
# Usage: bash run_interpretation.sh [target] [checkpoint_path] [hparams_path] [output_dir]
# Example: bash run_interpretation.sh rcf final_models_rcf/checkpoint.ckpt final_models_rcf/hparams.yaml results_interpret/rcf

TARGET=${1:-rcf}
CHECKPOINT=${2:-"final_models_rcf/trial_0/checkpoints/last.ckpt"}
HPARAMS=${3:-"final_models_rcf/trial_0/hparams.yaml"}
OUTPUT_DIR=${4:-"results_interpret/${TARGET}"}

echo "Starting Model Interpretation..."
echo "Target: $TARGET"
echo "Checkpoint: $CHECKPOINT"
echo "Hparams: $HPARAMS"
echo "Output Directory: $OUTPUT_DIR"

if [ "$TARGET" = "rcf" ]; then
    TRAIN_CSV="data/rcf_train_val.csv"
    TEST_X="data/interpret/rcf_test_X.csv"
    TEST_Y="data/interpret/rcf_test_y.csv"
    METADATA="metadata_rcf.json"
else
    TRAIN_CSV="data/tscf_train_val.csv"
    TEST_X="data/interpret/tscf_test_X.csv"
    TEST_Y="data/interpret/tscf_test_y.csv"
    METADATA="metadata_tscf.json"
fi

python scripts/04_interpret_models.py \
    --checkpoint "$CHECKPOINT" \
    --hparams "$HPARAMS" \
    --test_X "$TEST_X" \
    --test_y "$TEST_Y" \
    --train_X "$TRAIN_CSV" \
    --metadata_file "$METADATA" \
    --output_dir "$OUTPUT_DIR" \
    --target "$TARGET" \
    --mechanisms all \
    --n_baselines 10 \
    --seed 42

