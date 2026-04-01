# Carolina Ground Truth — NC Master Gardener Fine-Tune

Fine-tuned model specializing in North Carolina backyard gardening, micro-farming,
soil science, permaculture, and organic food production.

## Quick Start

### 1. Generate Training Data

```bash
cd /server/programming/pipeline
source activate.sh

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Optional: extract reference material from gardening ebooks
python gardener/extract_references.py /path/to/ebooks/ gardener/references/

# Generate training data (~3000 examples, ~$15-25 with Sonnet)
python gardener/generate.py --count 3000

# Or with ebook references for higher quality
python gardener/generate.py --count 3000 --references gardener/references/

# Dry run first to see what will be generated
python gardener/generate.py --dry-run
```

### 2. Train

```bash
bash gardener/train.sh

# Or with custom options (passed through to pipeline.py):
bash gardener/train.sh --no-export
```

### 3. Export & Quantize

The training script uses pipeline.py which handles LoRA merge + GGUF export.
For MagicQuant hybrid quantization:

```bash
python pipeline.py \
    --model unsloth/Qwen3-8B \
    --dataset gardener/gardener_training_data.jsonl \
    --output-dir ./output-gardener
```

## Architecture

- `system_prompt.py` — The "Carolina Ground Truth" persona definition
- `topics.py` — 39 subcategories across soil science, pest management, planting,
  water management, permaculture, seasonal planning, and more
- `generate.py` — Synthetic data generator using Claude API as teacher model
- `extract_references.py` — Ebook text extraction for reference-grounded generation
- `train.sh` — Training wrapper for the Unsloth pipeline

## Cost Estimate

With Claude Sonnet as teacher model:
- 3000 examples ≈ ~$15-25 (depending on response length)
- 5000 examples ≈ ~$25-40

## Using Reference Material

If you have gardening ebooks (PDF/EPUB), extract them first:

```bash
python gardener/extract_references.py ~/books/gardening/ gardener/references/
```

Then pass `--references gardener/references/` to the generator. ~30% of examples
will be grounded in reference material, producing more specific and authoritative
responses.
