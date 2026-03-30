#!/usr/bin/env python3
"""
LEGACY: Unsloth QLoRA Fine-Tuning Script

WARNING: This script uses Unsloth's FastLanguageModel which causes
single-threaded safetensors chunking on AMD APU unified memory. For 40B+
models on Strix Halo, use fast_train_zeroclaw.py instead (or pipeline.py
which now calls the fast loader automatically).

This script is kept for reference and for use on NVIDIA GPUs where Unsloth
works correctly.

Based on the approach used for peterjohannmedina/Medina-Qwen3.5-27B-OpenClaw.

Usage:
    python train.py --config config.yaml
    python train.py --model unsloth/Qwen3-8B --dataset ./my_data.jsonl --output ./output
"""

import os
import sys
import json
import argparse
from pathlib import Path

# AMD ROCm optimizations
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "backend:native,expandable_segments:True")
os.environ.setdefault("UNSLOTH_SKIP_TORCHVISION_CHECK", "1")

import torch
import yaml
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


# ---------------------------------------------------------------------------
# Default config (mirrors Medina-Qwen3.5-27B-OpenClaw hyperparameters)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Model
    "model_name": "unsloth/Qwen3-8B",
    "max_seq_length": 4096,
    "load_in_4bit": True,

    # LoRA
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "use_rslora": True,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # Training
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "fp16": False,
    "bf16": True,
    "logging_steps": 1,
    "save_strategy": "epoch",
    "seed": 42,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},

    # Dataset
    "dataset_path": None,          # Path to local JSONL/JSON or HF dataset name
    "dataset_split": "train",
    "dataset_text_field": "text",   # Field containing formatted text
    "chat_template": None,          # If set, apply chat template formatting

    # Output
    "output_dir": "./output",
    "hub_model_id": None,           # If set, push to HF Hub
    "save_merged": False,           # Save full merged model (not just adapters)
    "save_gguf": False,             # Also export GGUF quantizations
}


def load_config(config_path: str | None, cli_overrides: dict) -> dict:
    """Load config from YAML file and merge with CLI overrides."""
    config = dict(DEFAULT_CONFIG)

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            file_config = yaml.safe_load(f)
        if file_config:
            config.update(file_config)

    # CLI overrides take precedence
    for k, v in cli_overrides.items():
        if v is not None:
            config[k] = v

    return config


def format_chat_dataset(dataset, tokenizer, config):
    """Format dataset using the model's chat template.

    Expects each example to have a 'messages' field with the standard
    [{"role": "...", "content": "..."}] format. The chat template from
    the tokenizer is applied, and the result is stored in the text field.
    """
    text_field = config["dataset_text_field"]

    def apply_template(example):
        if "messages" in example:
            example[text_field] = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        return example

    return dataset.map(apply_template)


def load_training_dataset(config, tokenizer):
    """Load dataset from local file or HF Hub."""
    path = config["dataset_path"]
    if not path:
        raise ValueError("No dataset_path specified. Provide --dataset or set dataset_path in config.")

    p = Path(path)
    if p.exists():
        if p.suffix == ".jsonl":
            dataset = load_dataset("json", data_files=str(p), split="train")
        elif p.suffix == ".json":
            dataset = load_dataset("json", data_files=str(p), split="train")
        elif p.is_dir():
            dataset = load_dataset(str(p), split=config["dataset_split"])
        else:
            dataset = load_dataset("json", data_files=str(p), split="train")
    else:
        # Assume HF Hub dataset
        dataset = load_dataset(path, split=config["dataset_split"])

    # Apply chat template if messages field exists and chat_template is requested
    if config.get("chat_template") or "messages" in dataset.column_names:
        dataset = format_chat_dataset(dataset, tokenizer, config)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Unsloth QLoRA Fine-Tuning")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", type=str, dest="model_name", help="Model name or path")
    parser.add_argument("--dataset", type=str, dest="dataset_path", help="Dataset path (local or HF)")
    parser.add_argument("--output", type=str, dest="output_dir", help="Output directory")
    parser.add_argument("--epochs", type=int, dest="num_train_epochs", help="Number of epochs")
    parser.add_argument("--batch-size", type=int, dest="per_device_train_batch_size")
    parser.add_argument("--lr", type=float, dest="learning_rate")
    parser.add_argument("--lora-r", type=int, dest="lora_r", help="LoRA rank")
    parser.add_argument("--max-seq-len", type=int, dest="max_seq_length")
    parser.add_argument("--hub-id", type=str, dest="hub_model_id", help="HF Hub model ID to push to")
    parser.add_argument("--save-merged", action="store_true", dest="save_merged")
    parser.add_argument("--save-gguf", action="store_true", dest="save_gguf")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config" and k != "no_4bit"}
    if args.no_4bit:
        overrides["load_in_4bit"] = False

    config = load_config(args.config, overrides)

    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------------------------
    print(f"\nLoading model: {config['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
    )

    # -----------------------------------------------------------------------
    # 2. Attach LoRA adapters
    # -----------------------------------------------------------------------
    print("Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        use_rslora=config["use_rslora"],
        use_gradient_checkpointing="unsloth",
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # -----------------------------------------------------------------------
    # 3. Load dataset
    # -----------------------------------------------------------------------
    print(f"\nLoading dataset: {config['dataset_path']}")
    dataset = load_training_dataset(config, tokenizer)
    print(f"Dataset size: {len(dataset)} examples")

    # -----------------------------------------------------------------------
    # 4. Train
    # -----------------------------------------------------------------------
    output_dir = config["output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        optim=config["optim"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        seed=config["seed"],
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs=config.get("gradient_checkpointing_kwargs"),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field=config["dataset_text_field"],
        max_seq_length=config["max_seq_length"],
    )

    print("\nStarting training...")
    stats = trainer.train()
    print(f"\nTraining complete! Final loss: {stats.training_loss:.4f}")

    # -----------------------------------------------------------------------
    # 5. Save
    # -----------------------------------------------------------------------
    # Always save LoRA adapters
    lora_dir = os.path.join(output_dir, "lora_adapters")
    print(f"\nSaving LoRA adapters to {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    # Optionally save merged model
    if config["save_merged"]:
        merged_dir = os.path.join(output_dir, "merged_model")
        print(f"Saving merged model to {merged_dir}")
        model.save_pretrained_merged(merged_dir, tokenizer)

    # Optionally export GGUF
    if config["save_gguf"]:
        gguf_dir = os.path.join(output_dir, "gguf")
        print(f"Exporting GGUF to {gguf_dir}")
        model.save_pretrained_gguf(
            gguf_dir, tokenizer,
            quantization_method=["q4_k_m", "q5_k_m", "q8_0"],
        )

    # Optionally push to HF Hub
    if config["hub_model_id"]:
        print(f"Pushing to HF Hub: {config['hub_model_id']}")
        model.push_to_hub(config["hub_model_id"])
        tokenizer.push_to_hub(config["hub_model_id"])

    print("\nDone!")


if __name__ == "__main__":
    main()
