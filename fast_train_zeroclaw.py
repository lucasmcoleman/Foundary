#!/usr/bin/env python3
"""
Fast training script for AMD APU unified memory systems.

WHY THIS EXISTS:
On the Strix Halo APU (gfx1151), CPU and GPU share 128 GB of system RAM via GTT.
The default transformers from_pretrained() path loads tensors one at a time through
Python's GIL, taking hours for 40B+ models. Unsloth's memory handling made this
worse: as GTT filled, safetensors chunking became single-threaded and crawled.

This script bypasses both by:
1. Creating model skeleton on meta device (instant, zero memory)
2. Loading safetensors shards one at a time (fast buffered I/O)
3. Quantizing with BnB 4-bit on GPU in bulk per-shard (~0.01s/tensor)
4. Freeing each shard before loading the next (controls peak memory to ~30 GB)

Then trains with PEFT LoRA + TRL SFTTrainer with completion-only loss masking
(only assistant turns contribute to the loss; system/user turns are masked).
"""

import gc
import json
import os
import re
import time

# ROCm environment variables — must be set before importing torch.
# HSA_ENABLE_SDMA=0: Disables System DMA engine on AMD APUs, which can cause
#   hangs or data corruption on unified memory when GPU and CPU race for the bus.
# PYTORCH_HIP_ALLOC_CONF: Uses the native HIP allocator with expandable segments
#   so PyTorch can grow its memory pool without fragmentation.
# TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: Enables experimental AOTriton kernels
#   for better performance on RDNA/CDNA architectures.
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "backend:native,expandable_segments:True"
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
from pathlib import Path

MODEL_ID = "DavidAU/Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking"
DATASET_PATH = "zeroclaw_training_data.jsonl"
OUTPUT_DIR = "./output-zeroclaw-qwen40b"
HF_REPO = "lmcoleman/Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-Zeroclaw-GGUF"

# LoRA config
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training config
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 4096

DEVICE = torch.device("cuda:0")


def fast_load_quantized_model(model_id: str = MODEL_ID):
    """Load model with shard-by-shard 4-bit quantization on GPU.

    This is dramatically faster than transformers' default from_pretrained() on
    unified memory systems because:
    - Meta device creation allocates zero memory (just builds the module tree)
    - Each shard is loaded once via buffered I/O, quantized in bulk on GPU, then freed
    - BnB GPU quantization kernels run at ~0.01s/tensor (parallelized across all CUs)
    - Peak memory stays at ~30 GB for a 40B model vs 80+ GB for full-precision load

    Returns (model, tokenizer) with all Linear layers quantized to NF4 except
    embeddings, norms, and lm_head which remain in bfloat16.
    """
    from transformers import AutoConfig, AutoTokenizer
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    import bitsandbytes as bnb

    print(f"Loading config for {model_id}...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Ensuring model files are cached...")
    model_path = snapshot_download(model_id)
    print(f"Model path: {model_path}")

    # Load the safetensors index to know which tensors live in which shard file.
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    # Group tensors by shard so we can load one shard file at a time.
    shards = {}
    for tensor_name, shard_name in weight_map.items():
        shards.setdefault(shard_name, []).append(tensor_name)

    # Create model on meta device — this builds the full module tree (all layers,
    # attention heads, etc.) but allocates zero memory. Every parameter is a
    # meta tensor (no storage). We materialize real tensors shard-by-shard below.
    print("Creating model skeleton on meta device...")
    from transformers import AutoModelForCausalLM
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    model.eval()

    # BnB 4-bit quantization config.
    # NF4 (NormalFloat4) gives better quality than FP4 for LLM weights.
    quant_type = "nf4"
    compute_dtype = torch.bfloat16
    # blocksize=128 is REQUIRED on AMD GPUs. The default (64) causes silent
    # corruption because AMD's GPU quantization kernels expect 128-element blocks.
    blocksize = 128

    # Replace nn.Linear modules with bnb.nn.Linear4bit stubs.
    # Skip lm_head (needs full precision for output logits), embeddings, and norms.
    print("Replacing Linear layers with Linear4bit...")
    skip_names = {"lm_head"}

    def replace_linear_with_4bit(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Linear) and not any(s in full_name for s in skip_names):
                if not any(s in full_name for s in ["embed", "norm"]):
                    new_layer = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=compute_dtype,
                        quant_type=quant_type,
                        device="meta",
                    )
                    setattr(module, name, new_layer)
            else:
                replace_linear_with_4bit(child, full_name)

    replace_linear_with_4bit(model)

    # Load and quantize shard by shard.
    # Each iteration: load shard -> quantize all its Linear weights on GPU ->
    # place non-quantized tensors (embeds, norms) on GPU -> free shard -> next.
    # This keeps peak memory to roughly (largest shard) + (quantized model so far).
    total_t0 = time.time()
    loaded = 0
    total = len(weight_map)

    for shard_idx, (shard_name, tensor_names) in enumerate(sorted(shards.items())):
        shard_path = os.path.join(model_path, shard_name)
        shard_size = os.path.getsize(shard_path) / 1e9

        t0 = time.time()
        print(f"[{shard_idx+1}/{len(shards)}] Loading {shard_name} ({shard_size:.1f} GB, {len(tensor_names)} tensors)...")

        # Load entire shard at once — this is a single buffered read, much faster
        # than loading tensor-by-tensor through safetensors' Python interface.
        shard_data = load_file(shard_path, device="cpu")

        for name in tensor_names:
            # Navigate the module tree to find the target parameter.
            parts = name.split(".")
            target = model
            for part in parts[:-1]:
                target = getattr(target, part)
            attr = parts[-1]

            tensor = shard_data[name]

            if isinstance(target, bnb.nn.Linear4bit) and attr == "weight":
                # Quantize to NF4 on GPU. The .to(DEVICE) transfer and Params4bit
                # quantization both happen on GPU, utilizing all available CUs.
                tensor_gpu = tensor.to(DEVICE, dtype=compute_dtype)
                target.weight = bnb.nn.Params4bit(
                    tensor_gpu.contiguous(),
                    requires_grad=False,
                    quant_type=quant_type,
                    blocksize=blocksize,
                    compress_statistics=True,
                ).to(DEVICE)
                del tensor_gpu
            else:
                # Non-quantized tensors: embeddings, layer norms, biases.
                # These stay in bfloat16 (or their native dtype if non-float).
                dtype = compute_dtype if tensor.is_floating_point() else tensor.dtype
                new_param = torch.nn.Parameter(tensor.to(DEVICE, dtype=dtype), requires_grad=False)
                setattr(target, attr, new_param)

            del tensor
            loaded += 1

        # Free this shard's CPU buffer before loading the next one.
        del shard_data
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        total_elapsed = time.time() - total_t0
        print(f"  Done in {elapsed:.1f}s | Progress: {loaded}/{total} ({100*loaded/total:.0f}%) | Total: {total_elapsed:.0f}s")

    total_time = time.time() - total_t0
    gpu_mb = torch.cuda.memory_allocated() / 1e6
    print(f"\nModel loaded in {total_time:.0f}s | GPU allocated: {gpu_mb:.0f} MB")

    return model, tokenizer


def detect_response_template(tokenizer):
    """Auto-detect the assistant response template from the tokenizer's chat template.

    For completion-only loss masking, we need to identify the token sequence that
    marks the start of assistant responses. This is model-specific:
    - Qwen3.5: "<|im_start|>assistant\n"
    - Llama 3: "<|start_header_id|>assistant<|end_header_id|>\n\n"

    We render a minimal two-turn conversation and find the text between the last
    end-of-turn marker and the assistant's content. This works for any model with
    a standard chat template.
    """
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}],
        tokenize=False, add_generation_prompt=False,
    )

    y_pos = probe.rfind("Y")
    before_y = probe[:y_pos]

    # Strip <think>...</think> blocks that some models inject (e.g. Qwen3.5 thinking mode).
    before_y_clean = re.sub(r"<think>.*?</think>\s*", "", before_y, flags=re.DOTALL)

    # Find the last end-of-turn marker before the assistant content.
    end_markers = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>"]
    last_end = -1
    marker_len = 0
    for em in end_markers:
        p = before_y_clean.rfind(em)
        if p > last_end:
            last_end = p
            marker_len = len(em)

    if last_end >= 0:
        response_template = before_y_clean[last_end + marker_len:].lstrip("\n")
    else:
        # Fallback: use the last line before assistant content.
        response_template = before_y_clean.split("\n")[-1]

    print(f"Auto-detected response template: {repr(response_template)}")
    return response_template


def find_latest_checkpoint(output_dir: str):
    """Find the latest trainer checkpoint in the output directory for resume.

    HuggingFace Trainer saves checkpoints as checkpoint-<step> directories.
    Returns the path to the latest one, or None if no checkpoints exist.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = sorted(
        output_path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    if checkpoints:
        print(f"Found checkpoint: {checkpoints[-1]}")
        return str(checkpoints[-1])
    return None


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Stage 1: Fast model loading ──
    # Uses shard-by-shard quantization instead of transformers' default.
    model, tokenizer = fast_load_quantized_model()

    import psutil
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"Process RSS: {rss:.1f} GB")

    # ── Stage 2: Attach LoRA adapters ──
    # prepare_model_for_kbit_training freezes base weights and enables gradient
    # checkpointing so only LoRA parameters consume gradient memory.
    print("\nAttaching LoRA adapters...")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")

    # ── Stage 3: Load and format dataset ──
    print(f"\nLoading dataset: {DATASET_PATH}")
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Dataset: {len(dataset)} examples")

    # Apply the model's chat template to format messages into a single text string.
    def fmt(ex):
        ex["text"] = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False,
        )
        return ex

    dataset = dataset.map(fmt)

    # ── Stage 4: Set up completion-only loss masking ──
    # Only the assistant's responses contribute to the loss. System prompts and
    # user messages are masked so the model learns to generate responses, not
    # memorize prompts. This is critical for tool-call training (ZeroClaw) where
    # the system prompt is long and we want the model to learn the response format.
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

    response_template = detect_response_template(tokenizer)
    response_ids = tokenizer.encode(response_template, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_ids,
        tokenizer=tokenizer,
    )

    # ── Stage 5: Train ──
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=False,
        bf16=True,
        logging_steps=1,
        # Save at each epoch so training can resume if interrupted.
        save_strategy="epoch",
        seed=42,
        gradient_checkpointing=True,
        # use_reentrant=False is required for LoRA + gradient checkpointing to
        # work correctly. The reentrant implementation doesn't handle the mixed
        # frozen/trainable parameter graph that PEFT creates.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataset_text_field="text",
    )
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=collator,
    )

    # Resume from the latest checkpoint if one exists (e.g. after OOM or crash).
    resume_checkpoint = find_latest_checkpoint(OUTPUT_DIR)

    print("\nStarting training...")
    stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
    print(f"\nTraining complete! Final loss: {stats.training_loss:.4f}")

    # ── Stage 6: Save LoRA adapters ──
    # Only the LoRA adapter weights are saved (tiny compared to the full model).
    # fast_export.py can later merge these with the base model shard-by-shard.
    lora_dir = os.path.join(OUTPUT_DIR, "lora_adapters")
    print(f"\nSaving LoRA adapters to {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    print("PIPELINE_STAGE_COMPLETE=training")


if __name__ == "__main__":
    main()
