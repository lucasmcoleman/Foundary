#!/bin/bash
# Launch the Pipeline UI (FastAPI + WebSocket log streaming)
#
# Usage: ./run.sh [port]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${SCRIPT_DIR}/../unsloth-env/bin"

# Fallback to root-level venv
if [ ! -f "$VENV/uvicorn" ]; then
    VENV="/server/programming/pipeline/unsloth-env/bin"
fi

PORT="${1:-7865}"

export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True"
export UNSLOTH_SKIP_TORCHVISION_CHECK=1
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Starting Pipeline UI on http://localhost:${PORT}"
exec "$VENV/uvicorn" app:app --host 0.0.0.0 --port "$PORT" --app-dir "$SCRIPT_DIR"
