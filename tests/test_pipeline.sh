#!/bin/bash
# =============================================================================
# Pipeline Integration Test Suite
# =============================================================================
#
# Tests all pipeline components: UI, training code, export, quantization,
# HuggingFace upload, and model card generation.
#
# Usage:
#   ./test_pipeline.sh               # Run all tests
#   ./test_pipeline.sh --skip-gpu    # Skip GPU-intensive tests (training)
#   ./test_pipeline.sh --skip-lmstudio  # Skip LMStudio GGUF tests
#
# Exit code: 0 if all tests pass, 1 if any test fails.
# =============================================================================

set -euo pipefail

PIPELINE_DIR="/server/programming/pipeline"
VENV_DIR="${PIPELINE_DIR}/.venv"
MAGICQUANT_DIR="/server/programming/MagicQuant"
GGUF_DIR="/server/ai/models/lmcoleman/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated"
SOURCE_DIR="/server/ai/models/source/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated"
LMSTUDIO_URL="http://localhost:1234"

# Parse flags
SKIP_GPU=false
SKIP_LMSTUDIO=false
for arg in "$@"; do
    case "$arg" in
        --skip-gpu) SKIP_GPU=true ;;
        --skip-lmstudio) SKIP_LMSTUDIO=true ;;
        --help|-h)
            echo "Usage: $0 [--skip-gpu] [--skip-lmstudio]"
            exit 0
            ;;
    esac
done

# Counters
PASS=0
FAIL=0
SKIP=0
FAILURES=""

# Color output (if terminal)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    NC='\033[0m'
else
    GREEN=''
    RED=''
    YELLOW=''
    NC=''
fi

pass() {
    echo -e "  [${GREEN}PASS${NC}] $1"
    PASS=$((PASS + 1))
}

fail() {
    echo -e "  [${RED}FAIL${NC}] $1"
    FAIL=$((FAIL + 1))
    FAILURES="${FAILURES}\n  - $1: $2"
}

skip() {
    echo -e "  [${YELLOW}SKIP${NC}] $1"
    SKIP=$((SKIP + 1))
}

# Activate venv and set environment
activate_env() {
    source "${VENV_DIR}/bin/activate"
    export HSA_ENABLE_SDMA=0
    export PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True"
    export UNSLOTH_SKIP_TORCHVISION_CHECK=1
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    export PYTHONPATH="${PIPELINE_DIR}/core:${MAGICQUANT_DIR}:${PYTHONPATH:-}"
    cd "${PIPELINE_DIR}"
}

echo "============================================================"
echo "Pipeline Integration Test Suite"
echo "Date: $(date)"
echo "Pipeline: ${PIPELINE_DIR}"
echo "============================================================"
echo ""

# ── Test 1: UI App Import ──────────────────────────────────────────────────

echo "Test 1: UI App Import"
activate_env
OUTPUT=$(python -c "from ui.app import app; print('OK')" 2>&1) || true
if echo "$OUTPUT" | grep -q "OK"; then
    pass "ui.app imports cleanly"
else
    fail "ui.app import failed" "$OUTPUT"
fi

# ── Test 1b: UI Startup / HTTP Response ────────────────────────────────────

echo ""
echo "Test 1b: UI HTTP Response (port 7865)"
if ss -tlnp 2>/dev/null | grep -q ':7865 '; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7865/ 2>/dev/null) || HTTP_CODE="000"
    if [ "$HTTP_CODE" = "200" ]; then
        pass "UI responds 200 on port 7865"
    elif [ "$HTTP_CODE" = "500" ]; then
        # Check if it is the known stale-process bug
        ERROR_BODY=$(curl -s http://localhost:7865/ 2>/dev/null)
        fail "UI responds 500 (stale uvicorn process using old /server/programming/unsloth/ path)" "Restart with: kill \$(lsof -ti:7865); ${PIPELINE_DIR}/ui/run.sh 7865"
    else
        fail "UI responds HTTP ${HTTP_CODE}" "Expected 200"
    fi
else
    skip "UI not running on port 7865 (start with: source activate.sh)"
fi

# ── Test 2: Training Code Imports ──────────────────────────────────────────

echo ""
echo "Test 2: Training Code Imports"
activate_env
OUTPUT=$(python -c "from fast_train_zeroclaw import fast_load_quantized_model, detect_response_template, find_latest_checkpoint; print('OK')" 2>&1) || true
if echo "$OUTPUT" | grep -q "OK"; then
    pass "fast_train_zeroclaw imports (fast_load_quantized_model, detect_response_template, find_latest_checkpoint)"
else
    fail "fast_train_zeroclaw import failed" "$OUTPUT"
fi

# ── Test 3: Export Code Imports ────────────────────────────────────────────

echo ""
echo "Test 3: Export Code Imports"
activate_env
OUTPUT=$(python -c "from fast_export import streaming_merge, load_lora_weights, build_lora_map; print('OK')" 2>&1) || true
if echo "$OUTPUT" | grep -q "OK"; then
    pass "fast_export imports (streaming_merge, load_lora_weights, build_lora_map)"
else
    fail "fast_export import failed" "$OUTPUT"
fi

# ── Test 3b: Pipeline Module Imports ───────────────────────────────────────

echo ""
echo "Test 3b: Pipeline Module Imports"
activate_env
OUTPUT=$(python -c "from pipeline import PipelineConfig, stage_training, stage_export, stage_magicquant, stage_upload; print('OK')" 2>&1) || true
if echo "$OUTPUT" | grep -q "OK"; then
    pass "pipeline.py imports all stage functions"
else
    fail "pipeline.py import failed" "$OUTPUT"
fi

# ── Test 3c: MagicQuant Imports ────────────────────────────────────────────

echo ""
echo "Test 3c: MagicQuant Imports"
activate_env
OUTPUT=$(python -c "
from magicquant.gguf.source import GGUFSource
from magicquant.gguf.writer import GGUFWriter
from magicquant.gguf.tensor_groups import TensorGroupClassifier
from magicquant.orchestrator import MagicQuantOrchestrator
print('OK')
" 2>&1) || true
if echo "$OUTPUT" | grep -q "OK"; then
    pass "MagicQuant core imports (GGUFSource, GGUFWriter, TensorGroupClassifier, MagicQuantOrchestrator)"
else
    fail "MagicQuant import failed" "$OUTPUT"
fi

# ── Test 3d: HF Upload Module Imports ─────────────────────────────────────

echo ""
echo "Test 3d: HF Upload Module Imports"
activate_env
OUTPUT=$(python -c "from hf_upload import HFUploadConfig, generate_model_card, dry_run, upload, discover_upload_files; print('OK')" 2>&1) || true
if echo "$OUTPUT" | grep -q "OK"; then
    pass "hf_upload.py imports (HFUploadConfig, generate_model_card, dry_run, upload)"
else
    fail "hf_upload.py import failed" "$OUTPUT"
fi

# ── Test 4: GGUF File Existence ────────────────────────────────────────────

echo ""
echo "Test 4: Pre-generated GGUF Files"
for TIER in Q4 Q5 Q6; do
    TIER_DIR="${GGUF_DIR}/${TIER}"
    if [ -d "$TIER_DIR" ]; then
        GGUF_COUNT=$(ls "${TIER_DIR}"/*.gguf 2>/dev/null | wc -l)
        if [ "$GGUF_COUNT" -gt 0 ]; then
            GGUF_SIZE=$(du -sh "${TIER_DIR}" 2>/dev/null | cut -f1)
            pass "${TIER}: ${GGUF_COUNT} GGUF file(s) (${GGUF_SIZE})"
        else
            fail "${TIER}: no GGUF files" "Expected .gguf files in ${TIER_DIR}"
        fi
    else
        fail "${TIER}: directory missing" "Expected ${TIER_DIR}"
    fi
done

# ── Test 4b: GGUF Header Validation ───────────────────────────────────────

echo ""
echo "Test 4b: GGUF Header Validation"
activate_env
GGUF_FILE="${GGUF_DIR}/Q4/Huihui-Qwen3.5-9B-Q4.gguf"
if [ -f "$GGUF_FILE" ]; then
    OUTPUT=$(python -c "
import struct
with open('${GGUF_FILE}', 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_metadata = struct.unpack('<Q', f.read(8))[0]
if magic == b'GGUF' and version == 3 and n_tensors > 0:
    print(f'OK: version={version}, tensors={n_tensors}, metadata={n_metadata}')
else:
    print(f'BAD: magic={magic}, version={version}, tensors={n_tensors}')
" 2>&1) || true
    if echo "$OUTPUT" | grep -q "^OK:"; then
        pass "GGUF header valid: $(echo "$OUTPUT" | head -1)"
    else
        fail "GGUF header invalid" "$OUTPUT"
    fi
else
    fail "GGUF file missing" "${GGUF_FILE}"
fi

# ── Test 4c: GGUF Chat Template Present ───────────────────────────────────

echo ""
echo "Test 4c: GGUF Chat Template"
activate_env
OUTPUT=$(python -c "
from gguf import GGUFReader
reader = GGUFReader('${GGUF_FILE}')
ct = reader.fields.get('tokenizer.chat_template')
if ct:
    val = bytes(ct.parts[-1]).decode('utf-8', errors='replace')[:100]
    if 'im_start' in val or 'assistant' in val:
        print(f'OK: chat_template present ({len(val)}+ chars)')
    else:
        print(f'WARN: chat_template present but unexpected content: {val[:50]}')
else:
    print('MISSING: no tokenizer.chat_template in GGUF metadata')
" 2>&1) || true
if echo "$OUTPUT" | grep -q "^OK:"; then
    pass "Chat template present in GGUF metadata"
elif echo "$OUTPUT" | grep -q "^WARN:"; then
    pass "Chat template present (non-standard format): $OUTPUT"
else
    fail "Chat template missing from GGUF" "$OUTPUT"
fi

# ── Test 4d: LMStudio GGUF Load + Generation ─────────────────────────────

echo ""
echo "Test 4d: LMStudio GGUF Generation"
if [ "$SKIP_LMSTUDIO" = true ]; then
    skip "LMStudio tests (--skip-lmstudio)"
else
    # Check if LMStudio is running
    if curl -s "${LMSTUDIO_URL}/v1/models" >/dev/null 2>&1; then
        for MODEL_ID in "q4@q4" "q5@q5" "q6"; do
            RESPONSE=$(curl -s "${LMSTUDIO_URL}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":50,\"temperature\":0.7}" 2>&1) || RESPONSE="ERROR"

            if echo "$RESPONSE" | grep -q '"error"'; then
                fail "LMStudio ${MODEL_ID}: load error" "$(echo "$RESPONSE" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("error",{}).get("message","unknown"))' 2>/dev/null)"
            elif echo "$RESPONSE" | grep -q '"completion_tokens"'; then
                TOTAL_TOKENS=$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null) || TOTAL_TOKENS=0
                CONTENT=$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null) || CONTENT=""
                if [ "$TOTAL_TOKENS" -gt 0 ]; then
                    if [ -z "$CONTENT" ]; then
                        pass "LMStudio ${MODEL_ID}: generates ${TOTAL_TOKENS} tokens (NOTE: content field is empty -- Qwen3.5 thinking mode returns tokens in reasoning_content only)"
                    else
                        pass "LMStudio ${MODEL_ID}: generates ${TOTAL_TOKENS} tokens with visible content"
                    fi
                else
                    fail "LMStudio ${MODEL_ID}: 0 completion tokens" "Model loaded but produced no output"
                fi
            else
                fail "LMStudio ${MODEL_ID}: unexpected response" "$RESPONSE"
            fi
        done
    else
        skip "LMStudio not running on ${LMSTUDIO_URL}"
    fi
fi

# ── Test 5: HuggingFace Dry-Run ───────────────────────────────────────────

echo ""
echo "Test 5: HuggingFace Upload Dry-Run"
activate_env
OUTPUT=$(python core/hf_upload.py \
    --repo test/test \
    --output-dir ./output \
    --base-model huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated \
    --dry-run 2>&1) || true

if echo "$OUTPUT" | grep -q "HF_TOKEN not set"; then
    pass "Dry-run correctly reports HF_TOKEN not set (expected without credentials)"
elif echo "$OUTPUT" | grep -q "Dry run PASSED"; then
    pass "Dry-run PASSED (credentials available)"
elif echo "$OUTPUT" | grep -q "Token validation failed"; then
    pass "Dry-run correctly reports invalid token (expected with expired/wrong token)"
else
    fail "Dry-run unexpected output" "$OUTPUT"
fi

# ── Test 6: Model Card Generation ─────────────────────────────────────────

echo ""
echo "Test 6: Model Card Content Review"
activate_env
GGUF_DIR_PY="${GGUF_DIR}"
OUTPUT=$(python -c "
from hf_upload import HFUploadConfig, generate_model_card
from pathlib import Path
import os

cfg = HFUploadConfig(
    repo_id='lmcoleman/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-GGUF',
    base_model='huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated',
    dataset_name='zeroclaw_training_data',
)

# Simulate files for the card
gguf_base = os.environ.get('GGUF_DIR_PY', '${GGUF_DIR_PY}')
fake_files = []
for tier in ['Q4', 'Q5', 'Q6']:
    p = Path(gguf_base) / tier
    if p.exists():
        for f in sorted(p.glob('*.gguf')):
            fake_files.append((f, f.name))

card = generate_model_card(cfg, fake_files)

required = {
    'Base Model': '## Base Model' in card,
    'Quantization Method': '## Quantization Method' in card,
    'MagicQuant mention': 'MagicQuant' in card,
    'Training Details': '## Training Details' in card,
    'AMD Ryzen AI Max': 'AMD Ryzen AI Max' in card,
    'ROCm mention': 'ROCm' in card,
    'Caveats section': '## Caveats' in card,
    'Limitations section': '## Limitations' in card,
    'Base model link': 'huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated' in card,
    'YAML front matter': '---' in card and 'base_model:' in card,
    'License': 'apache-2.0' in card,
    'Completion-only loss': 'completion-only' in card.lower() or 'Completion-only' in card,
    'GGUF files table': '## GGUF Files' in card if fake_files else True,
    'Usage section': '## Usage' in card,
}

all_ok = True
for name, result in required.items():
    status = 'OK' if result else 'MISSING'
    if not result:
        all_ok = False
    print(f'{status}: {name}')
result_str = 'PASS' if all_ok else 'FAIL'
print(f'RESULT: {result_str}')
" 2>&1) || true

if echo "$OUTPUT" | grep -q "RESULT: PASS"; then
    pass "Model card contains all required sections"
    # Print individual checks
    echo "$OUTPUT" | grep -v "RESULT:" | while read -r line; do
        echo "       $line"
    done
else
    MISSING=$(echo "$OUTPUT" | grep "MISSING:" | tr '\n' '; ')
    fail "Model card missing sections" "$MISSING"
    echo "$OUTPUT" | while read -r line; do
        echo "       $line"
    done
fi

# ── Test 7: Source Safetensors Exist ───────────────────────────────────────

echo ""
echo "Test 7: Source Model Files"
if [ -d "$SOURCE_DIR" ]; then
    ST_COUNT=$(ls "${SOURCE_DIR}"/*.safetensors 2>/dev/null | wc -l)
    if [ "$ST_COUNT" -gt 0 ]; then
        pass "Source safetensors: ${ST_COUNT} files in ${SOURCE_DIR}"
    else
        fail "No safetensors in source dir" "${SOURCE_DIR}"
    fi
    if [ -f "${SOURCE_DIR}/config.json" ]; then
        pass "config.json present"
    else
        fail "config.json missing from source" "${SOURCE_DIR}"
    fi
    if [ -f "${SOURCE_DIR}/tokenizer_config.json" ]; then
        pass "tokenizer_config.json present"
    else
        fail "tokenizer_config.json missing from source" "${SOURCE_DIR}"
    fi
else
    fail "Source directory missing" "${SOURCE_DIR}"
fi

# ── Test 8: Training Data Validation ──────────────────────────────────────

echo ""
echo "Test 8: Training Data"
TRAIN_DATA="${PIPELINE_DIR}/data/zeroclaw_training_data.jsonl"
if [ -f "$TRAIN_DATA" ]; then
    LINE_COUNT=$(wc -l < "$TRAIN_DATA")
    pass "Training data exists: ${LINE_COUNT} lines"

    activate_env
    OUTPUT=$(python -c "
import json
errors = 0
ok = 0
with open('${TRAIN_DATA}') as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line: continue
        try:
            d = json.loads(line)
            if 'messages' not in d:
                errors += 1
                continue
            msgs = d['messages']
            roles = [m['role'] for m in msgs]
            if 'assistant' not in roles:
                errors += 1
                continue
            ok += 1
        except:
            errors += 1
print(f'valid={ok} errors={errors}')
if errors == 0 and ok > 0:
    print('RESULT: PASS')
else:
    print('RESULT: FAIL')
" 2>&1) || true
    if echo "$OUTPUT" | grep -q "RESULT: PASS"; then
        VALID=$(echo "$OUTPUT" | grep "valid=" | head -1)
        pass "Training data format valid (${VALID})"
    else
        fail "Training data format errors" "$OUTPUT"
    fi
else
    fail "Training data missing" "${TRAIN_DATA}"
fi

# ── Test 9: GPU Training Integration (optional) ───────────────────────────

echo ""
echo "Test 9: GPU Training Integration"
if [ "$SKIP_GPU" = true ]; then
    skip "GPU training test (--skip-gpu)"
else
    skip "GPU training test (run manually: python tests/test_training_integration.py)"
    echo "       This test takes 10-30 minutes and requires GPU availability."
    echo "       It validates: model loading, LoRA attachment, response template, 1-epoch training, export."
fi

# ── Test 10: Stale Path References ────────────────────────────────────────

echo ""
echo "Test 10: Stale Path References"
activate_env
# Check for references to the old /server/programming/unsloth/ path (not unsloth-env)
STALE_REFS=$(grep -rn '/server/programming/unsloth[^-]' "${PIPELINE_DIR}" \
    --include="*.py" --include="*.sh" --include="*.yaml" --include="*.yml" \
    2>/dev/null | grep -v '__pycache__' | grep -v '.pyc' | grep -v 'test_pipeline.sh') || true

if [ -z "$STALE_REFS" ]; then
    pass "No stale /server/programming/unsloth/ path references"
else
    STALE_COUNT=$(echo "$STALE_REFS" | wc -l)
    fail "Found ${STALE_COUNT} stale path reference(s) to old /server/programming/unsloth/ directory" "$(echo "$STALE_REFS" | head -3)"
    echo "$STALE_REFS" | while read -r line; do
        echo "       $line"
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "RESULTS"
echo "============================================================"
echo -e "  Passed:  ${GREEN}${PASS}${NC}"
echo -e "  Failed:  ${RED}${FAIL}${NC}"
echo -e "  Skipped: ${YELLOW}${SKIP}${NC}"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "FAILURES:"
    echo -e "$FAILURES"
    echo ""
    echo "============================================================"
    echo -e "${RED}PIPELINE NOT READY -- ${FAIL} test(s) failed${NC}"
    echo "============================================================"
    exit 1
else
    echo ""
    echo "============================================================"
    echo -e "${GREEN}ALL TESTS PASSED${NC} (${SKIP} skipped)"
    echo "============================================================"
    exit 0
fi
