#!/usr/bin/env bash
# Train the NC Master Gardener model using the existing pipeline.
#
# Usage:
#   cd /server/programming/pipeline
#   bash gardener/train.sh
#
# Prerequisites:
#   1. Generate training data first:  python gardener/generate.py
#   2. Activate venv:                 source activate.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="gardener/gardener_training_data.jsonl"
OUTPUT_DIR="./output-gardener"
MODEL="unsloth/Qwen3-8B"

# Validate dataset exists
if [ ! -f "$DATASET" ]; then
    echo "Error: Training data not found at $DATASET"
    echo "Run: python gardener/generate.py"
    exit 1
fi

EXAMPLES=$(wc -l < "$DATASET")
echo "Training NC Master Gardener model"
echo "  Dataset: $DATASET ($EXAMPLES examples)"
echo "  Base model: $MODEL"
echo "  Output: $OUTPUT_DIR"
echo ""

# Activate venv if not already
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source .venv/bin/activate
fi

python pipeline.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --no-magicquant \
    "$@"
