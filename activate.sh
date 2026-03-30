#!/bin/bash
# Activate the pipeline training environment
#
# Usage: source activate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate venv
source "$SCRIPT_DIR/unsloth-env/bin/activate"

# AMD ROCm optimizations for Strix Halo (gfx1151)
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True"
export UNSLOTH_SKIP_TORCHVISION_CHECK=1

# Faster HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Pipeline environment activated (Python $(python --version 2>&1 | cut -d' ' -f2))"
python -c "import torch; print(f'PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')" 2>/dev/null

# Start the Pipeline UI in the background
UI_PORT="${PIPELINE_UI_PORT:-7865}"
if ! ss -tlnp 2>/dev/null | grep -q ":${UI_PORT} "; then
    "$SCRIPT_DIR/ui/run.sh" "$UI_PORT" > "$SCRIPT_DIR/ui/ui.log" 2>&1 &
    PIPELINE_UI_PID=$!
    echo "Pipeline UI started on http://localhost:${UI_PORT} (pid $PIPELINE_UI_PID)"
else
    echo "Pipeline UI already running on http://localhost:${UI_PORT}"
fi
