"""
Foundry pipeline: Training → Export → Heretic → MagicQuant → HF Upload.

Uses custom fast loaders to avoid single-threaded safetensors chunking
that stalls on AMD APU unified memory (128 GB GTT).

Key design decisions:
  - Training uses shard-by-shard BnB 4-bit quantization on GPU (fast_train_zeroclaw.py)
  - Export uses streaming LoRA merge at ~6 GB peak memory (fast_export.py)
  - Heretic removes safety alignment via Optuna-optimized directional ablation
  - Completion-only loss masks system/user turns (only assistant responses contribute)
  - Dataset validation pre-flight check before committing GPU time
  - Auto-install llama.cpp if not found (needed by MagicQuant for perplexity probing)
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    model_name: str = "Tesslate/OmniCoder-9B"
    dataset_path: str = "data/zeroclaw_training_data.jsonl"
    max_seq_length: int = 8192
    load_in_4bit: bool = True  # Unused by fast loader (always 4-bit), kept for config compat
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_rslora: bool = True  # Unused by fast loader, kept for config compat
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    optim: str = "adamw_8bit"


@dataclass
class ExportConfig:
    """Single stage: merge LoRA + convert to GGUF."""
    gguf_type: str = "bf16"  # bf16, f16, q8_0, etc.
    also_save_merged: bool = False  # optionally also save HF safetensors


@dataclass
class HereticConfig:
    """Heretic abliteration (safety alignment removal via directional ablation)."""
    n_trials: int = 200
    n_startup_trials: int = 60
    quantization: str = "bnb_4bit"  # "none" or "bnb_4bit"
    kl_divergence_scale: float = 1.0
    orthogonalize_direction: bool = False
    row_normalization: str = "none"  # "none", "pre", or "full"


@dataclass
class MagicQuantConfig:
    target_base_quant: str = "MXFP4_MOE"
    generations: int = 50
    population_size: int = 100
    tiers: list[str] = field(default_factory=lambda: ["Q4", "Q5", "Q6"])
    verify: bool = False
    llamacpp_path: Optional[str] = None


@dataclass
class UploadConfig:
    repo_id: str = ""
    private: bool = True
    base_model: str = ""
    license: str = "apache-2.0"
    upload_lora: bool = False
    upload_merged: bool = False
    upload_gguf: bool = True


@dataclass
class PipelineConfig:
    output_dir: str = "./output"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: Optional[ExportConfig] = field(default_factory=ExportConfig)
    heretic: Optional[HereticConfig] = None
    magicquant: Optional[MagicQuantConfig] = field(default_factory=MagicQuantConfig)
    upload: Optional[UploadConfig] = None


# ── Artifact paths ───────────────────────────────────────────────────────────

class Artifacts:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def lora_dir(self) -> Path:
        return self.output_dir / "lora_adapters"

    @property
    def merged_dir(self) -> Path:
        return self.output_dir / "merged_model"

    @property
    def bf16_gguf(self) -> Path:
        return self.output_dir / "model-bf16.gguf"

    @property
    def heretic_dir(self) -> Path:
        return self.output_dir / "heretic_model"

    @property
    def magicquant_dir(self) -> Path:
        return self.output_dir / "magicquant"


# ── Helpers ──────────────────────────────────────────────────────────────────

LogFn = Callable[[str, str], None]


def _default_log(msg: str, level: str = "info"):
    prefix = {"error": "ERROR", "warn": "WARN", "success": "OK", "stage": ">>>"}.get(level, "   ")
    print(f"[{prefix}] {msg}")


def _run(cmd: list[str], log: LogFn, env_extra: dict = None, cwd: str = None) -> int:
    env = os.environ.copy()
    env.update({
        "HSA_ENABLE_SDMA": "0",
        "PYTORCH_HIP_ALLOC_CONF": "backend:native,expandable_segments:True",
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "PYTHONUNBUFFERED": "1",
    })
    if env_extra:
        env.update(env_extra)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, cwd=cwd, text=True, bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(line)
    proc.wait()
    return proc.returncode


def _find_python() -> str:
    return sys.executable


def _find_llamacpp(hint: Optional[str] = None) -> Optional[Path]:
    candidates = [
        hint,
        os.environ.get("LLAMACPP_PATH"),
        str(Path.home() / "llama.cpp"),
        "./llama.cpp",
        "/usr/local",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        for sub in [p / "convert_hf_to_gguf.py", p / "bin" / "convert_hf_to_gguf.py",
                     p / "build" / "bin" / "llama-perplexity"]:
            if sub.exists():
                return p
    return None


# ── Improvement #3: Dataset validation ───────────────────────────────────────

def validate_dataset(dataset_path: str, log: LogFn) -> bool:
    """Pre-flight check on the training dataset before committing GPU time."""
    log("Validating dataset", "stage")
    path = Path(dataset_path)

    if not path.exists():
        log(f"Dataset not found: {path}", "error")
        return False

    if path.stat().st_size == 0:
        log("Dataset file is empty", "error")
        return False

    errors = []
    warnings = []
    tool_calls = 0
    roles_seen = set()
    n = 0

    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON — {e}")
                if len(errors) >= 5:
                    errors.append("(stopping after 5 errors)")
                    break
                continue

            n += 1

            if "messages" not in ex:
                errors.append(f"Line {i}: missing 'messages' field")
                continue

            msgs = ex["messages"]
            if not isinstance(msgs, list) or len(msgs) < 2:
                errors.append(f"Line {i}: 'messages' must be a list with >= 2 entries")
                continue

            for j, msg in enumerate(msgs):
                if "role" not in msg or "content" not in msg:
                    errors.append(f"Line {i}, message {j}: missing 'role' or 'content'")
                    break
                roles_seen.add(msg["role"])

            # Count tool calls for coverage stats
            for msg in msgs:
                if msg.get("role") == "assistant" and "<tool_call>" in msg.get("content", ""):
                    tool_calls += 1

    if errors:
        for e in errors:
            log(f"  {e}", "error")
        log(f"Dataset validation failed with {len(errors)} errors", "error")
        return False

    # Warnings
    if n < 10:
        warnings.append(f"Only {n} examples — consider adding more for better generalization")
    if "system" not in roles_seen:
        warnings.append("No 'system' role found — model may not learn system prompt behavior")
    if "assistant" not in roles_seen:
        warnings.append("No 'assistant' role found — nothing for the model to learn")

    for w in warnings:
        log(f"  Warning: {w}", "warn")

    # Stats
    log(f"  Examples: {n}")
    log(f"  Roles: {sorted(roles_seen)}")
    log(f"  Tool call turns: {tool_calls}")
    log(f"  File size: {path.stat().st_size / 1024:.1f} KB")
    log("Dataset validation passed", "success")
    return True


# ── Improvement #4: Auto-install llama.cpp ───────────────────────────────────

def ensure_llamacpp(hint: Optional[str], log: LogFn) -> Optional[Path]:
    """Find llama.cpp, or auto-install it if missing."""
    found = _find_llamacpp(hint)
    if found:
        log(f"llama.cpp found at: {found}")
        return found

    install_dir = Path.home() / "llama.cpp"
    log("llama.cpp not found — auto-installing", "stage")

    # Clone
    log("Cloning llama.cpp from GitHub...")
    rc = _run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(install_dir)], log)
    if rc != 0:
        log("Failed to clone llama.cpp", "error")
        return None

    # Build
    log("Building llama.cpp (cmake)...")
    build_dir = install_dir / "build"
    rc = _run(["cmake", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release", str(install_dir)], log)
    if rc != 0:
        log("cmake configure failed", "error")
        return None

    import multiprocessing
    jobs = str(multiprocessing.cpu_count())
    rc = _run(["cmake", "--build", str(build_dir), "-j", jobs], log)
    if rc != 0:
        log("cmake build failed", "error")
        return None

    log(f"llama.cpp installed at: {install_dir}", "success")
    return install_dir


# ── Stage: Training (with completion-only loss) ─────────────────────────────
#
# Uses the custom fast loader (fast_train_zeroclaw.py) which creates the model
# on meta device, loads safetensors shard-by-shard with inline BnB 4-bit
# quantization, and uses PEFT LoRA directly. This avoids single-threaded
# safetensors chunking that crawls to a halt on unified memory as GTT fills.

def stage_training(config: PipelineConfig, artifacts: Artifacts, log: LogFn) -> bool:
    """Run QLoRA training with completion-only loss using the custom fast loader.

    Uses fast_train_zeroclaw.py's shard-by-shard loading to avoid
    single-threaded sequential safetensors chunking on AMD APU unified memory.
    """
    # Validate dataset before committing GPU time.
    if not validate_dataset(config.training.dataset_path, log):
        return False

    log("Starting QLoRA training (fast loader, completion-only loss)", "stage")
    tc = config.training

    # Generate a training script that uses the custom fast loader.
    # This runs as a subprocess so GPU memory is fully freed when it exits.
    script = f'''
import os, re, gc, json, time
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "backend:native,expandable_segments:True"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── Fast model loading (shard-by-shard with inline BnB quantization) ──
# Import and call the fast loader directly.
import sys
sys.path.insert(0, str(Path("{Path.cwd()}") / "core"))
from fast_train_zeroclaw import fast_load_quantized_model, find_latest_checkpoint

DEVICE = torch.device("cuda:0")
model, tokenizer = fast_load_quantized_model("{tc.model_name}")

# ── Attach LoRA adapters ──
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora_config = LoraConfig(
    r={tc.lora_r}, lora_alpha={tc.lora_alpha}, lora_dropout={tc.lora_dropout},
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {{trainable:,}} / {{total:,}} ({{100*trainable/total:.2f}}%)")

# ── Load and format dataset ──
dataset = load_dataset("json", data_files="{tc.dataset_path}", split="train")
print(f"Dataset: {{len(dataset)}} examples")

def fmt(ex):
    ex["text"] = tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False,
    )
    return ex
dataset = dataset.map(fmt)

# ── Training with completion-only loss ──
# Only assistant turns contribute to the loss. System/user turns are masked.
training_args = SFTConfig(
    output_dir="{config.output_dir}",
    num_train_epochs={tc.num_train_epochs},
    per_device_train_batch_size={tc.per_device_train_batch_size},
    gradient_accumulation_steps={tc.gradient_accumulation_steps},
    learning_rate={tc.learning_rate},
    lr_scheduler_type="{tc.lr_scheduler_type}",
    warmup_ratio={tc.warmup_ratio},
    optim="{tc.optim}",
    weight_decay=0.01, max_grad_norm=1.0,
    fp16=False, bf16=True,
    logging_steps=1, save_strategy="epoch", seed=42,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={{"use_reentrant": False}},
    report_to="none",
    max_length={tc.max_seq_length},
    dataset_text_field="text",
    completion_only_loss=True,
)

trainer = SFTTrainer(
    model=model, processing_class=tokenizer, train_dataset=dataset,
    args=training_args,
)

# Resume from checkpoint if one exists (e.g. after crash or OOM).
resume_ckpt = find_latest_checkpoint("{config.output_dir}")
stats = trainer.train(resume_from_checkpoint=resume_ckpt)
print(f"PIPELINE_TRAINING_LOSS={{stats.training_loss:.4f}}")

lora_dir = "{artifacts.lora_dir}"
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
print("PIPELINE_STAGE_COMPLETE=training")
'''

    script_path = artifacts.output_dir / "_stage_train.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)

    rc = _run([_find_python(), "-u", str(script_path)], log, cwd=str(Path.cwd()))
    if rc != 0:
        log(f"Training failed (exit code {rc})", "error")
        return False

    if not artifacts.lora_dir.exists():
        log("LoRA adapters directory not found after training", "error")
        return False

    log("Training complete", "success")
    return True


# ── Stage: Export (streaming LoRA merge) ──────────────────────────────────────
#
# Uses fast_export.py's streaming shard-by-shard merge instead of standard
# save_pretrained_merged(), which loads the entire model into memory (~80 GB
# for a 40B model). The streaming merge peaks at ~6 GB.

def stage_export(config: PipelineConfig, artifacts: Artifacts, log: LogFn) -> bool:
    """Merge LoRA adapters into base model using streaming shard-by-shard merge.

    Always produces safetensors output (MagicQuant or llama.cpp handles GGUF).
    Uses fast_export.py's streaming merge to avoid loading the full model.
    """
    ec = config.export

    if not artifacts.lora_dir.exists():
        log("No LoRA adapters found — run training first", "error")
        return False

    log("Merging LoRA to safetensors (streaming shard-by-shard)", "stage")

    # Read the adapter_config.json to find the base model ID.
    import json as _json
    adapter_config_path = artifacts.lora_dir / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_cfg = _json.load(f)
        base_model_id = adapter_cfg.get("base_model_name_or_path", config.training.model_name)
    else:
        base_model_id = config.training.model_name

    # Generate an export script that uses the custom fast merge.
    script = f'''
import os, sys
from pathlib import Path
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "backend:native,expandable_segments:True"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

sys.path.insert(0, str(Path("{Path.cwd()}") / "core"))
from fast_export import streaming_merge

streaming_merge(
    model_id="{base_model_id}",
    lora_dir="{artifacts.lora_dir}",
    merged_dir="{artifacts.merged_dir}",
)
print("PIPELINE_STAGE_COMPLETE=export")
'''

    script_path = artifacts.output_dir / "_stage_export.py"
    script_path.write_text(script)

    rc = _run([_find_python(), "-u", str(script_path)], log, cwd=str(Path.cwd()))
    if rc != 0:
        log(f"Export failed (exit code {rc})", "error")
        return False

    if artifacts.merged_dir.exists():
        log(f"Merged safetensors ready at {artifacts.merged_dir}", "success")
    else:
        log("Merged model directory not found after export", "error")
        return False

    log("Export complete", "success")
    return True


# ── Stage: Heretic (abliteration) ───────────────────────────────────────────
#
# Removes safety alignment from LLMs via Optuna-optimized directional ablation.
# Uses heretic-llm's internal API (not the CLI) to avoid interactive prompts.
# Runs as a subprocess so GPU memory is fully freed afterward.

def stage_heretic(config: PipelineConfig, artifacts: Artifacts, log: LogFn) -> bool:
    """Run heretic abliteration on the merged model.

    Invokes heretic's internal API programmatically (bypassing the interactive CLI)
    to run Optuna optimization, select the best Pareto-optimal trial, apply it,
    and save the merged abliterated model as HF safetensors.
    """
    log("Starting heretic abliteration", "stage")

    if not artifacts.merged_dir.exists():
        log("No merged model found — run export first", "error")
        return False

    hc = config.heretic
    heretic_output = str(artifacts.heretic_dir)
    checkpoint_dir = str(artifacts.output_dir / "_heretic_checkpoints")

    # Build a self-contained script that uses heretic's internal API directly.
    # This bypasses all questionary interactive prompts in heretic's main.py.
    script = f'''
import math
import os
import sys
import time
import warnings

os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "backend:native,expandable_segments:True"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
import torch.nn.functional as F
import optuna
import transformers
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna.exceptions import ExperimentalWarning
from optuna import Trial, TrialPruned
from dataclasses import asdict

from heretic.config import Settings, QuantizationMethod, RowNormalization
from heretic.model import Model, AbliterationParameters
from heretic.evaluator import Evaluator
from heretic.utils import (
    empty_cache,
    format_duration,
    get_trial_parameters,
    load_prompts,
    print_memory_usage,
)

# Override heretic's rich print with plain print for pipeline logging
import builtins
_real_print = builtins.print
from heretic import utils as _hu
_hu.print = _real_print

torch.set_grad_enabled(False)
torch._dynamo.config.cache_size_limit = 64
transformers.logging.set_verbosity_error()
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ExperimentalWarning)

model_path = "{artifacts.merged_dir}"
output_path = "{heretic_output}"
checkpoint_dir = "{checkpoint_dir}"

_real_print(f"Model: {{model_path}}")
_real_print(f"Output: {{output_path}}")

os.makedirs(checkpoint_dir, exist_ok=True)

# Create Settings programmatically (bypasses CLI arg parsing)
settings = Settings(
    model=model_path,
    quantization="{hc.quantization}",
    n_trials={hc.n_trials},
    n_startup_trials={hc.n_startup_trials},
    kl_divergence_scale={hc.kl_divergence_scale},
    orthogonalize_direction={hc.orthogonalize_direction},
    row_normalization="{hc.row_normalization}",
    study_checkpoint_dir=checkpoint_dir,
    batch_size=0,
)

model = Model(settings)
_real_print()
print_memory_usage()

_real_print()
_real_print(f"Loading good prompts from {{settings.good_prompts.dataset}}...")
good_prompts = load_prompts(settings, settings.good_prompts)
_real_print(f"  {{len(good_prompts)}} prompts loaded")

_real_print()
_real_print(f"Loading bad prompts from {{settings.bad_prompts.dataset}}...")
bad_prompts = load_prompts(settings, settings.bad_prompts)
_real_print(f"  {{len(bad_prompts)}} prompts loaded")

# Auto batch size
if settings.batch_size == 0:
    _real_print()
    _real_print("Determining optimal batch size...")
    batch_size = 1
    best_batch_size = -1
    best_performance = -1
    while batch_size <= settings.max_batch_size:
        _real_print(f"  Trying batch size {{batch_size}}... ", end="")
        prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
        prompts = prompts[:batch_size]
        try:
            model.get_responses(prompts)
            start_time = time.perf_counter()
            responses = model.get_responses(prompts)
            end_time = time.perf_counter()
        except Exception as error:
            if batch_size == 1:
                raise
            _real_print(f"Failed ({{error}})")
            break
        response_lengths = [len(model.tokenizer.encode(r)) for r in responses]
        performance = sum(response_lengths) / (end_time - start_time)
        _real_print(f"Ok ({{performance:.0f}} tokens/s)")
        if performance > best_performance:
            best_batch_size = batch_size
            best_performance = performance
        batch_size *= 2
    settings.batch_size = best_batch_size
    _real_print(f"  Chosen batch size: {{settings.batch_size}}")

# Check for common response prefix
_real_print()
_real_print("Checking for common response prefix...")
from os.path import commonprefix
responses = model.get_responses_batched(good_prompts[:100] + bad_prompts[:100])
model.response_prefix = commonprefix(responses).rstrip(" ")
if model.response_prefix.startswith("<think>"):
    model.response_prefix = "<think></think>"
elif model.response_prefix.startswith("<|channel|>analysis<|message|>"):
    model.response_prefix = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
elif model.response_prefix.startswith("<thought>"):
    model.response_prefix = "<thought></thought>"
elif model.response_prefix.startswith("[THINK]"):
    model.response_prefix = "[THINK][/THINK]"
if model.response_prefix:
    _real_print(f"  Prefix found: {{model.response_prefix!r}}")
else:
    _real_print("  None found")

evaluator = Evaluator(settings, model)

# Compute refusal directions
_real_print()
_real_print("Calculating per-layer refusal directions...")
_real_print("  Obtaining residuals for good prompts...")
good_residuals = model.get_residuals_batched(good_prompts)
_real_print("  Obtaining residuals for bad prompts...")
bad_residuals = model.get_residuals_batched(bad_prompts)

good_means = good_residuals.mean(dim=0)
bad_means = bad_residuals.mean(dim=0)
refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

if settings.orthogonalize_direction:
    good_directions = F.normalize(good_means, p=2, dim=1)
    projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
    refusal_directions = refusal_directions - projection_vector.unsqueeze(1) * good_directions
    refusal_directions = F.normalize(refusal_directions, p=2, dim=1)

del good_residuals, bad_residuals
empty_cache()

# Set up Optuna study
study_checkpoint_file = os.path.join(
    checkpoint_dir,
    "".join([(c if (c.isalnum() or c in ["_", "-"]) else "--") for c in settings.model]) + ".jsonl",
)
lock_obj = JournalFileOpenLock(study_checkpoint_file)
backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
storage = JournalStorage(backend)

trial_index = 0
start_index = 0
start_time = time.perf_counter()

def objective(trial):
    global trial_index
    trial_index += 1
    trial.set_user_attr("index", trial_index)

    direction_scope = trial.suggest_categorical("direction_scope", ["global", "per layer"])
    last_layer_index = len(model.get_layers()) - 1
    direction_index = trial.suggest_float("direction_index", 0.4 * last_layer_index, 0.9 * last_layer_index)
    if direction_scope == "per layer":
        direction_index = None

    parameters = {{}}
    for component in model.get_abliterable_components():
        max_weight = trial.suggest_float(f"{{component}}.max_weight", 0.8, 1.5)
        max_weight_position = trial.suggest_float(f"{{component}}.max_weight_position", 0.6 * last_layer_index, 1.0 * last_layer_index)
        min_weight = trial.suggest_float(f"{{component}}.min_weight", 0.0, 1.0)
        min_weight_distance = trial.suggest_float(f"{{component}}.min_weight_distance", 1.0, 0.6 * last_layer_index)
        parameters[component] = AbliterationParameters(
            max_weight=max_weight,
            max_weight_position=max_weight_position,
            min_weight=(min_weight * max_weight),
            min_weight_distance=min_weight_distance,
        )

    trial.set_user_attr("direction_index", direction_index)
    trial.set_user_attr("parameters", {{k: asdict(v) for k, v in parameters.items()}})

    _real_print(f"\\nRunning trial {{trial_index}} of {{settings.n_trials}}...")
    _real_print("  Resetting model...")
    model.reset_model()
    _real_print("  Abliterating...")
    model.abliterate(refusal_directions, direction_index, parameters)
    _real_print("  Evaluating...")
    score, kl_divergence, refusals = evaluator.get_score()

    elapsed = time.perf_counter() - start_time
    remaining = (elapsed / (trial_index - start_index)) * (settings.n_trials - trial_index)
    _real_print(f"  Elapsed: {{format_duration(elapsed)}}")
    if trial_index < settings.n_trials:
        _real_print(f"  Estimated remaining: {{format_duration(remaining)}}")

    trial.set_user_attr("kl_divergence", kl_divergence)
    trial.set_user_attr("refusals", refusals)
    return score

def objective_wrapper(trial):
    try:
        return objective(trial)
    except KeyboardInterrupt:
        trial.study.stop()
        raise TrialPruned()

study = optuna.create_study(
    sampler=TPESampler(n_startup_trials=settings.n_startup_trials, n_ei_candidates=128, multivariate=True),
    directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
    storage=storage,
    study_name="heretic",
    load_if_exists=True,
)
study.set_user_attr("settings", settings.model_dump_json())
study.set_user_attr("finished", False)

def count_completed():
    return sum(1 for t in study.trials if t.state == TrialState.COMPLETE)

start_index = trial_index = count_completed()
if start_index > 0:
    _real_print(f"\\nResuming existing study ({{start_index}} trials completed).")

remaining_trials = settings.n_trials - count_completed()
if remaining_trials > 0:
    study.optimize(objective_wrapper, n_trials=remaining_trials)

if count_completed() == settings.n_trials:
    study.set_user_attr("finished", True)

# Select best trial from Pareto front (lowest refusals, then lowest KL divergence)
completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
if not completed_trials:
    _real_print("No completed trials — abliteration failed")
    sys.exit(1)

sorted_trials = sorted(
    completed_trials,
    key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]),
)
min_divergence = math.inf
best_trials = []
for t in sorted_trials:
    kl = t.user_attrs["kl_divergence"]
    if kl < min_divergence:
        min_divergence = kl
        best_trials.append(t)

# Pick the trial with fewest refusals and acceptable KL divergence
best = best_trials[0]
_real_print(f"\\nSelected trial {{best.user_attrs['index']}}: "
            f"refusals={{best.user_attrs['refusals']}}, "
            f"KL divergence={{best.user_attrs['kl_divergence']:.4f}}")

# Apply the best trial
_real_print("Resetting model...")
model.reset_model()
_real_print("Abliterating with best parameters...")
model.abliterate(
    refusal_directions,
    best.user_attrs["direction_index"],
    {{k: AbliterationParameters(**v) for k, v in best.user_attrs["parameters"].items()}},
)

# Save the merged model
_real_print("Saving merged abliterated model...")
merged_model = model.get_merged_model()
merged_model.save_pretrained(output_path)
del merged_model
empty_cache()
model.tokenizer.save_pretrained(output_path)

_real_print(f"Abliterated model saved to {{output_path}}")
_real_print("PIPELINE_STAGE_COMPLETE=heretic")
'''

    script_path = artifacts.output_dir / "_stage_heretic.py"
    script_path.write_text(script)

    rc = _run([_find_python(), "-u", str(script_path)], log, cwd=str(Path.cwd()))
    if rc != 0:
        log(f"Heretic failed (exit code {rc})", "error")
        return False

    if not artifacts.heretic_dir.exists():
        log("Heretic output directory not found", "error")
        return False

    log(f"Abliterated model saved to {artifacts.heretic_dir}", "success")
    return True


# ── Stage: MagicQuant ────────────────────────────────────────────────────────

def stage_magicquant(config: PipelineConfig, artifacts: Artifacts, log: LogFn) -> bool:
    """Run MagicQuant evolutionary search and generate hybrid GGUFs.

    Reads from merged safetensors (preferred) or BF16 GGUF (fallback).
    """
    log("Starting MagicQuant evolutionary quantization", "stage")

    # Determine source: prefer heretic output, then merged safetensors, fall back to GGUF
    if artifacts.heretic_dir.exists():
        source_path = str(artifacts.heretic_dir)
        log(f"Source: abliterated model at {source_path}")
    elif artifacts.merged_dir.exists():
        source_path = str(artifacts.merged_dir)
        log(f"Source: merged safetensors at {source_path}")
    elif artifacts.bf16_gguf.exists():
        source_path = str(artifacts.bf16_gguf)
        log(f"Source: BF16 GGUF at {source_path}")
    else:
        log("No merged model or BF16 GGUF found — run export first", "error")
        return False

    mc = config.magicquant

    # Improvement #4: auto-install llama.cpp if needed
    llamacpp = ensure_llamacpp(mc.llamacpp_path, log)

    try:
        from magicquant.orchestrator import MagicQuantOrchestrator

        orch = MagicQuantOrchestrator(
            source_model_path=source_path,
            output_dir=str(artifacts.magicquant_dir),
            llamacpp_path=str(llamacpp) if llamacpp else None,
        )

        log(f"Search: generations={mc.generations}, population={mc.population_size}, base={mc.target_base_quant}")

        best_configs, tiered = orch.run_full_search(
            target_base_quant=mc.target_base_quant,
            max_generations=mc.generations,
            population_size=mc.population_size,
            verbose=True,
        )

        if not tiered:
            log("Evolutionary search produced no viable configurations", "error")
            return False

        log(f"Search complete — tiers: {list(tiered.keys())}")

        model_name = config.training.model_name.split("/")[-1]
        paths = orch.generate_tiered_models(
            tiered=tiered,
            model_name_prefix=model_name,
            tiers=mc.tiers,
            verify=mc.verify,
        )

        valid_paths = [p for p in paths if p]
        for p in valid_paths:
            size_gb = Path(p).stat().st_size / 1e9
            log(f"  {Path(p).name} ({size_gb:.1f} GB)")
        log(f"Generated {len(valid_paths)} hybrid GGUF files", "success")
        return True

    except Exception as e:
        log(f"MagicQuant error: {e}", "error")
        import traceback
        log(traceback.format_exc(), "error")
        return False


# ── Stage: Upload ────────────────────────────────────────────────────────────

def stage_upload(config: PipelineConfig, artifacts: Artifacts, log: LogFn) -> bool:
    """Upload artifacts to HuggingFace Hub.

    Delegates to hf_upload module for model card generation, progress
    reporting, and file upload. Supports dry-run mode via stage_upload_dry_run().
    """
    from hf_upload import HFUploadConfig, upload

    uc = config.upload
    if not uc or not uc.repo_id:
        log("No repo_id configured for upload", "error")
        return False

    tc = config.training
    hf_cfg = HFUploadConfig(
        repo_id=uc.repo_id,
        private=uc.private,
        license=uc.license,
        upload_gguf=uc.upload_gguf,
        upload_lora=uc.upload_lora,
        upload_merged=uc.upload_merged,
        base_model=uc.base_model or tc.model_name,
        dataset_name=tc.dataset_path,
        lora_r=tc.lora_r,
        lora_alpha=tc.lora_alpha,
        lora_dropout=tc.lora_dropout,
        num_epochs=tc.num_train_epochs,
        learning_rate=tc.learning_rate,
        max_seq_length=tc.max_seq_length,
        batch_size=tc.per_device_train_batch_size,
        gradient_accumulation=tc.gradient_accumulation_steps,
        optimizer=tc.optim,
        lr_scheduler=tc.lr_scheduler_type,
    )

    return upload(hf_cfg, config.output_dir, log=log)


def stage_upload_dry_run(config: PipelineConfig, artifacts: Artifacts, log: LogFn):
    """Dry-run upload: validate credentials and report what would be uploaded.

    Returns a DryRunReport (from hf_upload module).
    """
    from hf_upload import HFUploadConfig, dry_run

    uc = config.upload
    if not uc or not uc.repo_id:
        log("No repo_id configured for upload", "error")
        return None

    tc = config.training
    hf_cfg = HFUploadConfig(
        repo_id=uc.repo_id,
        private=uc.private,
        license=uc.license,
        upload_gguf=uc.upload_gguf,
        upload_lora=uc.upload_lora,
        upload_merged=uc.upload_merged,
        base_model=uc.base_model or tc.model_name,
        dataset_name=tc.dataset_path,
        lora_r=tc.lora_r,
        lora_alpha=tc.lora_alpha,
        lora_dropout=tc.lora_dropout,
        num_epochs=tc.num_train_epochs,
        learning_rate=tc.learning_rate,
        max_seq_length=tc.max_seq_length,
        batch_size=tc.per_device_train_batch_size,
        gradient_accumulation=tc.gradient_accumulation_steps,
        optimizer=tc.optim,
        lr_scheduler=tc.lr_scheduler_type,
    )

    return dry_run(hf_cfg, config.output_dir, log=log)


# ── Pipeline runner ──────────────────────────────────────────────────────────

STAGES = [
    ("training",   stage_training),
    ("export",     stage_export),
    ("heretic",    stage_heretic),
    ("magicquant", stage_magicquant),
    ("upload",     stage_upload),
]


def run_pipeline(config: PipelineConfig, log: LogFn = _default_log) -> dict[str, bool]:
    """Run the full pipeline. Returns {stage_name: success/None(skipped)}."""
    artifacts = Artifacts(config.output_dir)
    results = {}

    enabled = set()
    if config.training is not None:
        enabled.add("training")
    if config.export is not None:
        enabled.add("export")
    if config.heretic is not None:
        enabled.add("heretic")
    if config.magicquant is not None:
        enabled.add("magicquant")
    if config.upload is not None:
        enabled.add("upload")

    log(f"Pipeline: {' → '.join(s for s, _ in STAGES if s in enabled)}", "stage")

    for stage_name, stage_fn in STAGES:
        if stage_name not in enabled:
            log(f"Skipping {stage_name}")
            results[stage_name] = None
            continue

        ok = stage_fn(config, artifacts, log)
        results[stage_name] = ok
        if not ok:
            log(f"Pipeline stopped at {stage_name}", "error")
            break

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training + Quantization Pipeline")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--heretic", action="store_true", help="Enable heretic abliteration stage")
    parser.add_argument("--no-heretic", action="store_true", help="Disable heretic abliteration stage")
    parser.add_argument("--heretic-trials", type=int, default=200, help="Heretic Optuna trials")
    parser.add_argument("--heretic-quantization", type=str, default="bnb_4bit",
                        help="Heretic model quantization (none or bnb_4bit)")
    parser.add_argument("--no-magicquant", action="store_true")
    parser.add_argument("--upload-to", type=str, help="HF repo ID")
    parser.add_argument("--llamacpp-path", type=str)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate upload credentials and show what would be uploaded (no actual upload)")
    args = parser.parse_args()

    cfg = PipelineConfig(output_dir=args.output_dir)

    if args.config:
        import yaml
        with open(args.config) as f:
            data = yaml.safe_load(f)
        if "training" in data:
            for k, v in data["training"].items():
                setattr(cfg.training, k, v)

    if args.model:
        cfg.training.model_name = args.model
    if args.dataset:
        cfg.training.dataset_path = args.dataset
    if args.no_export:
        cfg.export = None
    if args.heretic and not args.no_heretic:
        cfg.heretic = HereticConfig(
            n_trials=args.heretic_trials,
            quantization=args.heretic_quantization,
        )
    if args.no_heretic:
        cfg.heretic = None
    if args.no_magicquant:
        cfg.magicquant = None
    if args.upload_to:
        cfg.upload = UploadConfig(repo_id=args.upload_to)
    if args.llamacpp_path and cfg.magicquant:
        cfg.magicquant.llamacpp_path = args.llamacpp_path

    # Dry-run mode: validate upload without running the full pipeline
    if args.dry_run:
        if not cfg.upload:
            if args.upload_to:
                cfg.upload = UploadConfig(repo_id=args.upload_to)
            else:
                print("ERROR: --dry-run requires --upload-to <repo_id>")
                sys.exit(1)
        artifacts = Artifacts(cfg.output_dir)
        report = stage_upload_dry_run(cfg, artifacts, _default_log)
        sys.exit(0 if report and report.ok else 1)

    results = run_pipeline(cfg)

    print("\n" + "=" * 50)
    print("Pipeline Results:")
    for stage, ok in results.items():
        sym = "+" if ok else ("-" if ok is None else "X")
        print(f"  {sym} {stage}")
