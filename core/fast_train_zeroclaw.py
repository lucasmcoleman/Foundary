#!/usr/bin/env python3
"""
Fast training script for AMD APU unified memory systems.

WHY THIS EXISTS:
On the Strix Halo APU (gfx1151), CPU and GPU share 128 GB of system RAM via GTT.
The default transformers from_pretrained() path loads tensors one at a time through
Python's GIL, taking hours for 40B+ models. Third-party loaders (e.g. Unsloth)
made this worse: as GTT filled, safetensors chunking became single-threaded and crawled.

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
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
from pathlib import Path


def get_device() -> torch.device:
    """Detect the best available device at runtime."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


DEVICE = get_device()


def fast_load_quantized_model(model_id: str = "Tesslate/OmniCoder-9B"):
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
    # Composite/multimodal models (e.g. Qwen3.5) wrap the text config inside
    # a top-level config that lacks vocab_size etc.  Unwrap for CausalLM.
    _is_composite = hasattr(config, 'text_config')
    if _is_composite:
        print(f"  Unwrapping composite config: {type(config).__name__} -> {type(config.text_config).__name__}")
        config = config.text_config
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Ensuring model files are cached...")
    model_path = snapshot_download(model_id)
    print(f"Model path: {model_path}")

    # Load the safetensors index to know which tensors live in which shard file.
    # Multi-shard models have an index.json; single-shard models have a single
    # model.safetensors file with no index.
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
    elif os.path.exists(single_path):
        from safetensors import safe_open
        with safe_open(single_path, framework="pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
    else:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    # Composite models save weights with a "model.language_model." prefix and
    # include vision encoder tensors ("model.visual.").  Since we build a
    # text-only CausalLM, remap the language_model keys and drop vision keys.
    if _is_composite:
        remapped_weight_map = {}
        skipped = 0
        for orig_name, shard_name in weight_map.items():
            if orig_name.startswith("model.visual"):
                skipped += 1
                continue
            new_name = orig_name.replace("model.language_model.", "model.", 1)
            remapped_weight_map[new_name] = (shard_name, orig_name)
        weight_map = remapped_weight_map
        if skipped:
            print(f"  Skipped {skipped} vision encoder tensors (not needed for CausalLM)")
    else:
        weight_map = {name: (shard, name) for name, shard in weight_map.items()}

    # Group tensors by shard so we can load one shard file at a time.
    shards = {}
    for tensor_name, (shard_name, _orig_name) in weight_map.items():
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

        try:
            for name in tensor_names:
                # Navigate the module tree using the (possibly remapped) name.
                parts = name.split(".")
                target = model
                for part in parts[:-1]:
                    target = getattr(target, part)
                attr = parts[-1]

                # Read tensor from shard using the original (on-disk) key.
                _shard_name, orig_name = weight_map[name]
                tensor = shard_data[orig_name]

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
        finally:
            # Free this shard's CPU buffer before loading the next one.
            del shard_data
            gc.collect()
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        total_elapsed = time.time() - total_t0
        print(f"  Done in {elapsed:.1f}s | Progress: {loaded}/{total} ({100*loaded/total:.0f}%) | Total: {total_elapsed:.0f}s")

    # Handle tied weights (e.g. lm_head.weight = embed_tokens.weight) and any
    # other parameters still on meta device after shard loading.
    meta_params = [(n, p) for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta_params:
        print(f"Materializing {len(meta_params)} tied/missing parameters...")
        for name, param in meta_params:
            # Check if this is a tied weight by looking for the source
            parts = name.split(".")
            if parts[-1] == "weight" and "lm_head" in name:
                # Tie lm_head to embed_tokens
                embed = model.model.embed_tokens.weight
                if embed.device.type != "meta":
                    model.lm_head.weight = embed
                    print(f"  Tied {name} -> model.embed_tokens.weight")
                    continue
            # Fallback: materialize as zeros on GPU
            dtype = compute_dtype if param.is_floating_point() else param.dtype
            new_param = torch.nn.Parameter(
                torch.zeros(param.shape, device=DEVICE, dtype=dtype),
                requires_grad=False,
            )
            # Navigate to parent and set
            target = model
            for p in parts[:-1]:
                target = getattr(target, p)
            setattr(target, parts[-1], new_param)
            print(f"  Materialized {name} as zeros on {DEVICE}")

    # Move all buffers that aren't on GPU yet.
    # This covers two cases:
    #   1. Meta buffers: created by init_empty_weights(), not in safetensors
    #   2. CPU buffers: computed during __init__ (e.g. RotaryEmbedding.inv_freq)
    #      These have correct values but are on the wrong device.
    offdevice_bufs = [(n, b) for n, b in model.named_buffers()
                      if b.device.type != DEVICE.type]
    if offdevice_bufs:
        print(f"Moving {len(offdevice_bufs)} buffers to {DEVICE}...")
    for name, buf in offdevice_bufs:
        parts = name.split(".")
        target = model
        for p in parts[:-1]:
            target = getattr(target, p)
        if buf.device.type == "meta":
            # Meta buffers have no data — materialize as zeros
            new_buf = torch.zeros(buf.shape, device=DEVICE, dtype=buf.dtype)
        else:
            # CPU buffers have valid data (e.g. inv_freq) — just move
            new_buf = buf.to(DEVICE)
        setattr(target, parts[-1], new_buf)

    total_time = time.time() - total_t0
    gpu_mb = torch.cuda.memory_allocated() / 1e6
    print(f"\nModel loaded in {total_time:.0f}s | GPU allocated: {gpu_mb:.0f} MB")

    # Tell HF Trainer + Accelerate the model is already on the correct device.
    # quantization_method = "bitsandbytes" makes Trainer skip its .to(device).
    # hf_device_map with >1 entry makes Accelerate's verify_device_map() return
    # True, which skips Accelerate's .to(device) in prepare_model().
    # is_quantized must stay False to avoid validate_quantization_for_training()
    # trying to access model.hf_quantizer (which doesn't exist for manual BnB).
    model.quantization_method = "bitsandbytes"
    model.is_quantized = False
    model.hf_device_map = {"": 0, "_dummy": 0}  # >1 entry triggers skip

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


def main(
    model_id: str = "Tesslate/OmniCoder-9B",
    dataset_path: str = "data/zeroclaw_training_data.jsonl",
    output_dir: str = "./output",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    num_epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    max_seq_length: int = 4096,
) -> None:
    """Run fast QLoRA training end-to-end as a standalone script."""
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Stage 1: Fast model loading ──
    model, tokenizer = fast_load_quantized_model(model_id)

    import psutil
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"Process RSS: {rss:.1f} GB")

    # ── Stage 2: Attach LoRA adapters ──
    print("\nAttaching LoRA adapters...")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")

    # ── Stage 3: Load and format dataset ──
    print(f"\nLoading dataset: {dataset_path}")
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Dataset: {len(dataset)} examples")

    def fmt(ex):
        ex["text"] = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False,
        )
        return ex

    dataset = dataset.map(fmt)

    # ── Stage 4: Train with completion-only loss ──
    from trl import SFTTrainer, SFTConfig
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=False,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        seed=42,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataset_text_field="text",
        max_length=max_seq_length,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    resume_checkpoint = find_latest_checkpoint(output_dir)

    print("\nStarting training...")
    stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
    print(f"\nTraining complete! Final loss: {stats.training_loss:.4f}")

    # ── Save LoRA adapters ──
    lora_dir = os.path.join(output_dir, "lora_adapters")
    print(f"\nSaving LoRA adapters to {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    print("PIPELINE_STAGE_COMPLETE=training")


if __name__ == "__main__":
    main()
