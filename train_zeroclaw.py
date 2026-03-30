#!/usr/bin/env python3
"""
LEGACY: Train OmniCoder-9B on ZeroClaw tool-call data using Unsloth QLoRA.

WARNING: This script uses Unsloth's FastLanguageModel which causes
single-threaded safetensors chunking on AMD APU unified memory. For
training on Strix Halo, use fast_train_zeroclaw.py instead.
"""

import os

# AMD ROCm optimizations
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "backend:native,expandable_segments:True"
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_NAME = "Tesslate/OmniCoder-9B"
DATASET_PATH = "zeroclaw_training_data.jsonl"
OUTPUT_DIR = "./output-zeroclaw"

MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

# LoRA config (matching Medina-Qwen3.5 approach)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
USE_RSLORA = True
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training config
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4  # effective batch = 8
LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.05
OPTIM = "adamw_8bit"

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. Load model
    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # 2. Attach LoRA
    print("Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        use_rslora=USE_RSLORA,
        use_gradient_checkpointing="unsloth",
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # 3. Load and format dataset
    print(f"\nLoading dataset: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def format_example(example):
        """Apply chat template to messages."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_example)
    print(f"Dataset: {len(dataset)} examples")

    # Show a sample
    sample = dataset[0]["text"]
    print(f"Sample (first 500 chars):\n{sample[:500]}...\n")

    # 4. Train
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        optim=OPTIM,
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
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("Starting training...")
    stats = trainer.train()
    print(f"\nTraining complete! Final loss: {stats.training_loss:.4f}")

    # 5. Save LoRA adapters
    lora_dir = os.path.join(OUTPUT_DIR, "lora_adapters")
    print(f"\nSaving LoRA adapters to {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
