#!/usr/bin/env python3
"""
Fast LoRA merge + export for AMD APU unified memory systems.

WHY THIS EXISTS:
Standard save_pretrained_merged() loads the entire model into memory to merge
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


def get_device() -> torch.device:
    """Detect the best available device at runtime.

    Uses GPU for LoRA merge math (matmul B @ A). On unified memory this is
    effectively free since the tensors are in the same physical RAM, but the
    GPU can parallelize the matmul across all compute units.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


DEVICE = get_device()


def load_lora_weights(lora_dir: str):
    """Load LoRA adapter config and weights.

    LoRA adapters are small (~100 MB for r=32 on a 40B model) so we load them
    all into GPU memory upfront. They stay resident while we stream through
    the base model shards.
    """
    config_path = os.path.join(lora_dir, "adapter_config.json")
    with open(config_path) as f:
        lora_config = json.load(f)

    # PEFT can save adapter weights as either safetensors or .bin (PyTorch).
    safetensors_path = os.path.join(lora_dir, "adapter_model.safetensors")
    bin_path = os.path.join(lora_dir, "adapter_model.bin")
    # Load directly to GPU — LoRA weights are small and we need them for
    # every shard that contains a target module.
    if os.path.exists(safetensors_path):
        lora_weights = load_file(safetensors_path, device=str(DEVICE))
    elif os.path.exists(bin_path):
        lora_weights = torch.load(bin_path, map_location=DEVICE, weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter weights found in {lora_dir}")

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


def streaming_merge(
    model_id: str = "Tesslate/OmniCoder-9B",
    lora_dir: str = "./output/lora_adapters",
    merged_dir: str = "./output/merged_model",
) -> None:
    """Merge LoRA adapters into base model shard-by-shard.

    For each shard:
    1. Load the shard from disk (buffered I/O)
    2. For each weight that has a LoRA adapter, compute delta on GPU:
       W_new = W + (alpha/r) * B @ A
    3. Save the merged shard to the output directory
    4. Free the shard before loading the next

    This keeps peak memory to ~6 GB (one shard + LoRA weights).

    If lora_dir is None, copies the base model shards without merging (useful
    when exporting a base model without LoRA adapters).
    """
    from huggingface_hub import snapshot_download

    if lora_dir is not None:
        print(f"Loading LoRA from {lora_dir}")
        lora_config, lora_weights = load_lora_weights(lora_dir)
        lora_map = build_lora_map(lora_config, lora_weights)
    else:
        print("No LoRA adapters — copying base model shards directly")
        lora_map = {}

    print(f"\nEnsuring base model cached: {model_id}")
    model_path = snapshot_download(model_id)

    # Multi-shard models have an index.json; single-shard models have a single
    # model.safetensors file with no index.
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")
    idx_metadata = {}
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
        idx_metadata = idx.get("metadata", {})
    elif os.path.exists(single_path):
        from safetensors import safe_open
        with safe_open(single_path, framework="pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
    else:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

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
        src = os.path.join(lora_dir, fname) if lora_dir is not None else None
        if src and os.path.exists(src):
            shutil.copy2(src, merged_path / fname)
        else:
            src2 = os.path.join(model_path, fname)
            if os.path.exists(src2):
                shutil.copy2(src2, merged_path / fname)

    # Detect composite naming: LoRA keys use CausalLM names (model.layers.*)
    # but composite model shards use model.language_model.layers.*.
    # Build a mapping from shard tensor names to LoRA map keys.
    _composite_prefix = "model.language_model."
    _has_composite_names = any(k.startswith(_composite_prefix) for k in weight_map)
    if _has_composite_names and lora_map:
        print("  Detected composite model naming — will remap LoRA keys during merge")

    merged_count = 0
    total_t0 = time.time()
    new_weight_map = {}

    for shard_idx, (shard_name, tensor_names) in enumerate(sorted(shards.items())):
        shard_path = os.path.join(model_path, shard_name)
        shard_size = os.path.getsize(shard_path) / 1e9

        t0 = time.time()
        print(f"[{shard_idx+1}/{len(shards)}] Processing {shard_name} ({shard_size:.1f} GB, {len(tensor_names)} tensors)...")

        shard_data = load_file(shard_path, device="cpu")

        try:
            modified = 0
            for name in tensor_names:
                # For composite models, strip the language_model prefix to match
                # LoRA keys (PEFT stores keys without the composite wrapper).
                lora_key = name
                if _has_composite_names and name.startswith(_composite_prefix):
                    lora_key = "model." + name[len(_composite_prefix):]

                if lora_key in lora_map:
                    a_weight, b_weight, scaling = lora_map[lora_key]
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

            # Save merged shard to output directory. Write to a temp file first
            # then atomically rename, so a crash mid-write can't leave a corrupt shard.
            out_path = merged_path / shard_name
            tmp_path = out_path.with_suffix(".tmp")
            save_file(shard_data, str(tmp_path))
            tmp_path.rename(out_path)
        finally:
            # Free shard memory before loading the next one.
            del shard_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({modified} LoRA merges)")

    # Write the safetensors index so HF tooling can find the shards.
    new_index = {
        "metadata": idx_metadata,
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
