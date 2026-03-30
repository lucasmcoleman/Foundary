#!/usr/bin/env python3
"""
Fast LoRA merge + export for AMD APU unified memory systems.

WHY THIS EXISTS:
Unsloth's save_pretrained_merged() loads the entire model into memory to merge
LoRA weights — 80+ GB for a 40B model. On the Strix Halo APU this exhausts the
shared 128 GB GTT pool and forces swapping to disk.

This script streams through safetensors shards one at a time, applying LoRA
deltas (W_new = W + scaling * B @ A) inline on GPU, and writes merged shards
to disk. Peak memory: ~6 GB (one shard + LoRA weights in memory at a time).

The output is a standard HuggingFace safetensors model directory that can be
fed directly to MagicQuant for GGUF quantization or loaded by any HF-compatible
tool.
"""

import gc
import json
import os
import shutil
import time

import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

MODEL_ID = "DavidAU/Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking"
LORA_DIR = "./output-zeroclaw-qwen40b/lora_adapters"
MERGED_DIR = "./output-zeroclaw-qwen40b/merged_model"

# Use GPU for LoRA merge math (matmul B @ A). On unified memory this is
# effectively free since the tensors are in the same physical RAM, but the
# GPU can parallelize the matmul across all compute units.
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_lora_weights(lora_dir: str):
    """Load LoRA adapter config and weights.

    LoRA adapters are small (~100 MB for r=32 on a 40B model) so we load them
    all into GPU memory upfront. They stay resident while we stream through
    the base model shards.
    """
    config_path = os.path.join(lora_dir, "adapter_config.json")
    with open(config_path) as f:
        lora_config = json.load(f)

    weights_path = os.path.join(lora_dir, "adapter_model.safetensors")
    # Load directly to GPU — LoRA weights are small and we need them for
    # every shard that contains a target module.
    lora_weights = load_file(weights_path, device=str(DEVICE))

    print(f"LoRA config: r={lora_config['r']}, alpha={lora_config['lora_alpha']}")
    print(f"Target modules: {lora_config['target_modules']}")
    print(f"LoRA tensors: {len(lora_weights)}")

    return lora_config, lora_weights


def build_lora_map(lora_config, lora_weights):
    """Build a map from base model weight name -> (lora_A, lora_B, scaling).

    LoRA decomposes weight updates as: delta_W = (alpha/r) * B @ A
    where A is (r, in_features) and B is (out_features, r).
    The scaling factor alpha/r controls the magnitude of the update.
    """
    r = lora_config["r"]
    alpha = lora_config["lora_alpha"]
    scaling = alpha / r

    lora_map = {}

    # PEFT names LoRA weights like:
    #   base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    #   base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
    # We pair them and map back to the base model weight name.
    lora_a_keys = {k: v for k, v in lora_weights.items() if ".lora_A." in k}

    for a_key, a_weight in lora_a_keys.items():
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        if b_key not in lora_weights:
            print(f"  WARNING: No lora_B for {a_key}")
            continue

        b_weight = lora_weights[b_key]

        # Strip PEFT prefix and LoRA suffix to get the base model key:
        # "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        #  -> "model.layers.0.self_attn.q_proj.weight"
        base_key = a_key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")

        lora_map[base_key] = (a_weight, b_weight, scaling)

    print(f"LoRA merge targets: {len(lora_map)} weight matrices")
    return lora_map


def streaming_merge(model_id: str = MODEL_ID, lora_dir: str = LORA_DIR,
                    merged_dir: str = MERGED_DIR):
    """Merge LoRA adapters into base model shard-by-shard.

    For each shard:
    1. Load the shard from disk (buffered I/O)
    2. For each weight that has a LoRA adapter, compute delta on GPU:
       W_new = W + (alpha/r) * B @ A
    3. Save the merged shard to the output directory
    4. Free the shard before loading the next

    This keeps peak memory to ~6 GB (one shard + LoRA weights).
    """
    from huggingface_hub import snapshot_download

    print(f"Loading LoRA from {lora_dir}")
    lora_config, lora_weights = load_lora_weights(lora_dir)
    lora_map = build_lora_map(lora_config, lora_weights)

    print(f"\nEnsuring base model cached: {model_id}")
    model_path = snapshot_download(model_id)

    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    # Group tensors by shard file for sequential processing.
    shards = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)

    merged_path = Path(merged_dir)
    merged_path.mkdir(parents=True, exist_ok=True)

    # Copy model config files from the base model.
    for fname in ["config.json", "generation_config.json", "special_tokens_map.json"]:
        src = os.path.join(model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, merged_path / fname)

    # Tokenizer files: prefer LoRA dir (training may have modified tokenizer_config),
    # fall back to base model.
    for fname in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        src = os.path.join(lora_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, merged_path / fname)
        else:
            src2 = os.path.join(model_path, fname)
            if os.path.exists(src2):
                shutil.copy2(src2, merged_path / fname)

    merged_count = 0
    total_t0 = time.time()
    new_weight_map = {}

    for shard_idx, (shard_name, tensor_names) in enumerate(sorted(shards.items())):
        shard_path = os.path.join(model_path, shard_name)
        shard_size = os.path.getsize(shard_path) / 1e9

        t0 = time.time()
        print(f"[{shard_idx+1}/{len(shards)}] Processing {shard_name} ({shard_size:.1f} GB, {len(tensor_names)} tensors)...")

        shard_data = load_file(shard_path, device="cpu")

        modified = 0
        for name in tensor_names:
            if name in lora_map:
                a_weight, b_weight, scaling = lora_map[name]
                orig_dtype = shard_data[name].dtype

                # Perform the LoRA merge on GPU: W_new = W + scaling * (B @ A)
                # The matmul B @ A is the expensive part and benefits from GPU
                # parallelism. On unified memory the transfer cost is near zero.
                w = shard_data[name].to(DEVICE, dtype=torch.float32)
                delta = scaling * (b_weight.float() @ a_weight.float())
                shard_data[name] = (w + delta).to(dtype=orig_dtype).cpu()
                modified += 1
                merged_count += 1
                del w, delta

            new_weight_map[name] = shard_name

        # Save merged shard to output directory.
        out_path = merged_path / shard_name
        save_file(shard_data, str(out_path))

        # Free shard memory before loading the next one.
        del shard_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({modified} LoRA merges)")

    # Write the safetensors index so HF tooling can find the shards.
    new_index = {
        "metadata": idx.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    with open(merged_path / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    total_time = time.time() - total_t0
    print(f"\nMerge complete in {total_time:.0f}s | {merged_count} weights merged | Output: {merged_dir}")


def main():
    streaming_merge()
    print("PIPELINE_STAGE_COMPLETE=export")


if __name__ == "__main__":
    main()
