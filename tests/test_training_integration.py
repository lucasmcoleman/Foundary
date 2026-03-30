#!/usr/bin/env python3
"""
Lightweight integration test for the custom fast training pipeline.

Runs 1 epoch of training on zeroclaw_training_data.jsonl against
huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated (a 9B model
that fits in memory without hitting GTT limits during testing).

Validates:
1. fast_load_quantized_model() loads without error
2. LoRA adapters attach correctly
3. Completion-only loss masking is set up (response template detected)
4. 1 epoch of training produces a valid checkpoint
5. LoRA adapters are saved in a format fast_export.py can consume
6. fast_export.py can produce a merged safetensors directory

Usage:
    python tests/test_training_integration.py

NOTE: This test requires GPU access and downloads a 9B model (~5 GB).
      Do not run while the GPU is busy (e.g. during GGUF generation).
      Expected runtime: ~10-30 minutes depending on dataset size.
"""

import gc
import json
import os
import sys
import tempfile
import traceback

# Set ROCm environment before any torch import.
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "backend:native,expandable_segments:True"
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# Add the pipeline root to path so we can import the fast loaders.
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)

import torch

# Test configuration — use the 9B model for manageable test times.
TEST_MODEL_ID = "huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated"
DATASET_PATH = os.path.join(PIPELINE_ROOT, "zeroclaw_training_data.jsonl")

# Use a temp directory for test output so we don't pollute the workspace.
TEST_OUTPUT_DIR = tempfile.mkdtemp(prefix="pipeline_test_")


def test_model_loading():
    """Test that fast_load_quantized_model loads the 9B model successfully."""
    print("\n=== Test 1: Model Loading ===")
    from fast_train_zeroclaw import fast_load_quantized_model

    model, tokenizer = fast_load_quantized_model(TEST_MODEL_ID)

    # Verify model is on GPU.
    first_param = next(model.parameters())
    assert first_param.device.type == "cuda", f"Model not on GPU: {first_param.device}"

    # Verify tokenizer works.
    tokens = tokenizer.encode("Hello, world!")
    assert len(tokens) > 0, "Tokenizer produced empty output"

    print(f"  Model loaded on {first_param.device}")
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    return model, tokenizer


def test_lora_attachment(model):
    """Test that LoRA adapters attach correctly."""
    print("\n=== Test 2: LoRA Attachment ===")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=16,  # Smaller rank for faster test
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    assert trainable > 0, "No trainable parameters after LoRA attachment"
    assert trainable < total_params, "All parameters are trainable (LoRA not applied)"

    pct = 100 * trainable / total_params
    print(f"  Trainable: {trainable:,} / {total_params:,} ({pct:.2f}%)")
    assert pct < 5, f"Trainable percentage too high ({pct:.2f}%), LoRA may not be working"
    return model


def test_response_template(tokenizer):
    """Test that response template auto-detection works for Qwen3.5."""
    print("\n=== Test 3: Response Template Detection ===")
    from fast_train_zeroclaw import detect_response_template

    template = detect_response_template(tokenizer)
    assert template, "Response template is empty"
    assert len(template) > 0, "Response template has zero length"

    # For Qwen3.5 models, we expect something like "<|im_start|>assistant\n"
    print(f"  Detected template: {repr(template)}")

    # Verify it tokenizes to a non-empty sequence.
    ids = tokenizer.encode(template, add_special_tokens=False)
    assert len(ids) > 0, "Response template tokenizes to empty sequence"
    print(f"  Template token IDs: {ids}")
    return template


def test_training_one_epoch(model, tokenizer):
    """Test that 1 epoch of training completes and saves a checkpoint."""
    print("\n=== Test 4: Training (1 epoch) ===")
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
    from fast_train_zeroclaw import detect_response_template

    # Load dataset.
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"  Dataset: {len(dataset)} examples")

    def fmt(ex):
        ex["text"] = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False,
        )
        return ex
    dataset = dataset.map(fmt)

    # Set up completion-only loss masking.
    response_template = detect_response_template(tokenizer)
    response_ids = tokenizer.encode(response_template, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_ids,
        tokenizer=tokenizer,
    )

    training_args = SFTConfig(
        output_dir=TEST_OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Small for fast test
        learning_rate=2e-4,
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
    )
    tokenizer.model_max_length = 4096

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=collator,
    )

    stats = trainer.train()
    loss = stats.training_loss
    print(f"  Training loss: {loss:.4f}")
    assert loss > 0, "Training loss is zero — something is wrong"
    assert loss < 100, f"Training loss unreasonably high: {loss}"

    # Verify checkpoint was saved.
    from pathlib import Path
    checkpoints = list(Path(TEST_OUTPUT_DIR).glob("checkpoint-*"))
    assert len(checkpoints) > 0, "No checkpoints saved after training"
    print(f"  Checkpoints: {[c.name for c in checkpoints]}")

    return model, tokenizer


def test_lora_save(model, tokenizer):
    """Test that LoRA adapters save correctly."""
    print("\n=== Test 5: LoRA Save ===")
    lora_dir = os.path.join(TEST_OUTPUT_DIR, "lora_adapters")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    # Verify required files exist.
    required = ["adapter_config.json", "adapter_model.safetensors"]
    for fname in required:
        fpath = os.path.join(lora_dir, fname)
        assert os.path.exists(fpath), f"Missing required file: {fname}"
        size = os.path.getsize(fpath)
        print(f"  {fname}: {size / 1e6:.1f} MB")
        assert size > 0, f"File is empty: {fname}"

    # Verify adapter_config.json is valid JSON with expected fields.
    with open(os.path.join(lora_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    assert "r" in cfg, "adapter_config.json missing 'r'"
    assert "lora_alpha" in cfg, "adapter_config.json missing 'lora_alpha'"
    assert "target_modules" in cfg, "adapter_config.json missing 'target_modules'"
    print(f"  Config: r={cfg['r']}, alpha={cfg['lora_alpha']}")

    return lora_dir


def test_export(lora_dir):
    """Test that fast_export.py can merge LoRA adapters with the base model."""
    print("\n=== Test 6: LoRA Merge (fast_export) ===")
    from fast_export import streaming_merge

    merged_dir = os.path.join(TEST_OUTPUT_DIR, "merged_model")

    streaming_merge(
        model_id=TEST_MODEL_ID,
        lora_dir=lora_dir,
        merged_dir=merged_dir,
    )

    # Verify output exists and has safetensors files.
    from pathlib import Path
    merged_path = Path(merged_dir)
    assert merged_path.exists(), "Merged directory not created"

    st_files = list(merged_path.glob("*.safetensors"))
    assert len(st_files) > 0, "No safetensors files in merged output"
    print(f"  Safetensors files: {len(st_files)}")

    # Verify index file.
    idx_path = merged_path / "model.safetensors.index.json"
    assert idx_path.exists(), "Missing model.safetensors.index.json"
    with open(idx_path) as f:
        idx = json.load(f)
    assert "weight_map" in idx, "Index missing weight_map"
    print(f"  Weight map entries: {len(idx['weight_map'])}")

    # Verify config was copied.
    assert (merged_path / "config.json").exists(), "Missing config.json in merged output"
    print("  config.json present")

    total_size = sum(f.stat().st_size for f in st_files) / 1e9
    print(f"  Total merged size: {total_size:.1f} GB")


def run_all_tests():
    """Run all integration tests in sequence."""
    print("=" * 60)
    print("Training Pipeline Integration Test")
    print(f"Model: {TEST_MODEL_ID}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {TEST_OUTPUT_DIR}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = []

    try:
        model, tokenizer = test_model_loading()
        passed += 1
    except Exception as e:
        failed += 1
        errors.append(("Model Loading", traceback.format_exc()))
        print(f"  FAILED: {e}")
        return passed, failed, errors  # Cannot continue without model

    try:
        model = test_lora_attachment(model)
        passed += 1
    except Exception as e:
        failed += 1
        errors.append(("LoRA Attachment", traceback.format_exc()))
        print(f"  FAILED: {e}")
        return passed, failed, errors

    try:
        test_response_template(tokenizer)
        passed += 1
    except Exception as e:
        failed += 1
        errors.append(("Response Template", traceback.format_exc()))
        print(f"  FAILED: {e}")

    try:
        model, tokenizer = test_training_one_epoch(model, tokenizer)
        passed += 1
    except Exception as e:
        failed += 1
        errors.append(("Training", traceback.format_exc()))
        print(f"  FAILED: {e}")
        return passed, failed, errors

    try:
        lora_dir = test_lora_save(model, tokenizer)
        passed += 1
    except Exception as e:
        failed += 1
        errors.append(("LoRA Save", traceback.format_exc()))
        print(f"  FAILED: {e}")
        return passed, failed, errors

    # Free model memory before export test.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    try:
        test_export(lora_dir)
        passed += 1
    except Exception as e:
        failed += 1
        errors.append(("Export", traceback.format_exc()))
        print(f"  FAILED: {e}")

    return passed, failed, errors


if __name__ == "__main__":
    passed, failed, errors = run_all_tests()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, tb in errors:
            print(f"\n--- {name} ---")
            print(tb)
    print("=" * 60)

    # Clean up temp directory.
    import shutil
    print(f"\nTest output at: {TEST_OUTPUT_DIR}")
    print("Run 'rm -rf {TEST_OUTPUT_DIR}' to clean up.")

    sys.exit(1 if failed > 0 else 0)
