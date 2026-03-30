#!/usr/bin/env python3
"""
Pipeline UI — FastAPI backend.

Orchestrates a 4-stage LLM fine-tuning pipeline:
  Training → Export GGUF → MagicQuant → Upload

Uses WebSocket for real-time log streaming to the browser.
Port defaults to 7865 (configurable via PIPELINE_UI_PORT env var).
"""

import asyncio
import json
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

app = FastAPI(title="Pipeline UI")

PIPELINE_DIR = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(PIPELINE_DIR / "unsloth-env" / "bin" / "python")
if not Path(VENV_PYTHON).exists():
    VENV_PYTHON = "/server/programming/pipeline/unsloth-env/bin/python"


# ── State ────────────────────────────────────────────────────────────────────

class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"

ALL_STAGES = ["training", "export", "magicquant", "upload"]

class PipelineState:
    """Shared mutable state for the running pipeline, including WebSocket fan-out."""

    def __init__(self):
        self.stages = {s: StageStatus.PENDING for s in ALL_STAGES}
        self.running = False
        self.current_stage = None
        self.progress = 0
        self.ws_clients: list[WebSocket] = []

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.ws_clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.remove(ws)

    async def log(self, text: str, level: str = "info"):
        await self.broadcast({"type": "log", "text": text, "level": level, "ts": time.time()})

    async def set_stage(self, stage: str, status: StageStatus):
        self.stages[stage] = status
        if status == StageStatus.RUNNING:
            self.current_stage = stage
        await self.broadcast({"type": "stage_update", "stage": stage, "status": status.value})

    async def set_progress(self, pct: int):
        self.progress = pct
        await self.broadcast({"type": "progress", "percent": pct})

state = PipelineState()


# ── Pydantic models ──────────────────────────────────────────────────────────

class TrainingCfg(BaseModel):
    model_name: str = "Tesslate/OmniCoder-9B"
    dataset_path: str = "zeroclaw_training_data.jsonl"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_rslora: bool = True
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    optim: str = "adamw_8bit"
    output_dir: str = "./output"

class ExportCfg(BaseModel):
    gguf_type: str = "bf16"
    also_save_merged: bool = False
    source_model: str = ""  # when training is skipped: HF ID or local path to model/lora

class MagicQuantCfg(BaseModel):
    target_base_quant: str = "MXFP4_MOE"
    generations: int = 50
    population_size: int = 100
    tiers: list[str] = ["Q4", "Q5", "Q6"]
    llamacpp_path: str = ""
    source_model: str = ""  # when export is skipped: path to GGUF or merged model dir

class UploadCfg(BaseModel):
    repo_id: str = ""
    private: bool = True
    license: str = "apache-2.0"
    upload_gguf: bool = True
    upload_lora: bool = False
    upload_merged: bool = False

class RunRequest(BaseModel):
    training: TrainingCfg = TrainingCfg()
    export: Optional[ExportCfg] = ExportCfg()
    magicquant: Optional[MagicQuantCfg] = None
    upload: Optional[UploadCfg] = None
    enabled_stages: list[str] = ["training", "export"]


# ── Subprocess helper ────────────────────────────────────────────────────────

async def run_script(script: str, output_dir: str) -> int:
    """Write a Python script to disk and execute it in the venv, streaming stdout to WebSocket clients."""
    # Resolve relative paths against the project root, not uvicorn's CWD
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = PIPELINE_DIR / out_path
    script_path = out_path / f"_stage_{int(time.time())}.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)

    env = os.environ.copy()
    env.update({
        "HSA_ENABLE_SDMA": "0",
        "PYTORCH_HIP_ALLOC_CONF": "backend:native,expandable_segments:True",
        "UNSLOTH_SKIP_TORCHVISION_CHECK": "1",
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "PYTHONUNBUFFERED": "1",
    })

    proc = await asyncio.create_subprocess_exec(
        VENV_PYTHON, "-u", str(script_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env, cwd=str(PIPELINE_DIR),
        limit=1024 * 1024,  # 1MB line buffer — tqdm \r bars can be huge
    )

    async for raw in proc.stdout:
        # tqdm uses \r without \n, so one "line" may contain many \r-separated updates.
        # Split on \r and process the last (most recent) segment.
        segments = raw.decode("utf-8", errors="replace").split("\r")
        for text in segments:
            text = text.strip()
            if not text:
                continue
            if "it/s]" in text or "s/it]" in text:
                try:
                    pct = int(float(text.split("%|")[0].strip().split()[-1]))
                    await state.set_progress(pct)
                except (ValueError, IndexError):
                    pass
                continue  # don't log every progress bar update
            if "'loss'" in text:
                await state.log(text, "metric")
            elif "PIPELINE_STAGE_COMPLETE" in text:
                await state.log(text, "success")
            elif "Error" in text or "error" in text.lower():
                await state.log(text, "error")
            else:
                await state.log(text)

    await proc.wait()
    return proc.returncode


# ── Dataset validation (Improvement #3) ──────────────────────────────────────

async def validate_dataset(path: str) -> bool:
    """Pre-flight dataset check."""
    await state.log("Validating dataset...", "stage")
    p = Path(path)
    if not p.exists():
        await state.log(f"Dataset not found: {path}", "error")
        return False
    if p.stat().st_size == 0:
        await state.log("Dataset file is empty", "error")
        return False

    errors = []
    n = 0
    tool_calls = 0
    roles = set()

    with open(p) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON — {e}")
                if len(errors) >= 5:
                    break
                continue
            n += 1
            if "messages" not in ex:
                errors.append(f"Line {i}: missing 'messages' field")
                continue
            msgs = ex["messages"]
            if not isinstance(msgs, list) or len(msgs) < 2:
                errors.append(f"Line {i}: 'messages' needs >= 2 entries")
                continue
            for msg in msgs:
                if "role" not in msg or "content" not in msg:
                    errors.append(f"Line {i}: message missing 'role' or 'content'")
                    break
                roles.add(msg["role"])
                if msg["role"] == "assistant" and "<tool_call>" in msg.get("content", ""):
                    tool_calls += 1

    if errors:
        for e in errors[:5]:
            await state.log(f"  {e}", "error")
        await state.log(f"Validation failed ({len(errors)} errors)", "error")
        return False

    if n < 10:
        await state.log(f"  Warning: only {n} examples", "warn")
    await state.log(f"  {n} examples, {tool_calls} tool-call turns, roles: {sorted(roles)}")
    await state.log("Dataset valid", "success")
    return True


# ── Stage runners ────────────────────────────────────────────────────────────

def _resolve_out(output_dir: str) -> Path:
    """Resolve output_dir the same way run_script does."""
    p = Path(output_dir)
    return p if p.is_absolute() else PIPELINE_DIR / p


async def do_training(cfg: RunRequest) -> bool:
    """Run the QLoRA training stage. Skips if LoRA adapters already exist."""
    tc = cfg.training
    out = _resolve_out(tc.output_dir)

    # Skip if LoRA adapters already exist
    adapter_cfg = out / "lora_adapters" / "adapter_config.json"
    if adapter_cfg.exists():
        await state.log(f"LoRA adapters already exist at {out / 'lora_adapters'} — skipping training", "success")
        await state.set_stage("training", StageStatus.COMPLETE)
        await state.set_progress(100)
        return True

    # Improvement #3: validate first
    if not await validate_dataset(tc.dataset_path):
        return False

    await state.set_stage("training", StageStatus.RUNNING)
    await state.set_progress(0)
    await state.log("Starting QLoRA training (completion-only loss)", "stage")

    # Improvement #2: completion_only_loss
    script = f'''
import os
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "backend:native,expandable_segments:True"
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

from pathlib import Path as _P
_model_id = "{tc.model_name}"
_is_local = _P(_model_id).exists()
if _is_local:
    print(f"Loading from local path: {{_model_id}}")
else:
    try:
        from huggingface_hub import scan_cache_dir
        _cached = False
        for _repo in scan_cache_dir().repos:
            if _repo.repo_id == _model_id:
                _size_gb = _repo.size_on_disk / 1e9
                print(f"Model found in HF cache ({{_size_gb:.1f}} GB) — no download needed")
                _cached = True
                break
        if not _cached:
            print(f"Model not in cache — will download from HuggingFace: {{_model_id}}")
    except Exception:
        print(f"Loading model: {{_model_id}}")

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{tc.model_name}",
    max_seq_length={tc.max_seq_length},
    load_in_4bit={tc.load_in_4bit},
)
model = FastLanguageModel.get_peft_model(model,
    r={tc.lora_r}, lora_alpha={tc.lora_alpha}, lora_dropout={tc.lora_dropout},
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_rslora={tc.use_rslora}, use_gradient_checkpointing="unsloth",
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {{trainable:,}} / {{total:,}} ({{100*trainable/total:.2f}}%)")

dataset = load_dataset("json", data_files="{tc.dataset_path}", split="train")
print(f"Dataset: {{len(dataset)}} examples")

def fmt(ex):
    ex["text"] = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
    return ex
dataset = dataset.map(fmt)

# Resolve the actual tokenizer — some models (e.g. VL models) return a Processor
# which wraps the tokenizer. SFTTrainer needs the real tokenizer for completion masking.
_tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
_use_completion_only = hasattr(_tok, "encode")
if _use_completion_only:
    print("Completion-only loss: enabled (masking non-assistant tokens)")
else:
    print("Completion-only loss: disabled (processor lacks encode method)")

trainer = SFTTrainer(
    model=model, processing_class=_tok, train_dataset=dataset,
    args=SFTConfig(
        output_dir="{tc.output_dir}",
        num_train_epochs={tc.num_train_epochs},
        per_device_train_batch_size={tc.per_device_train_batch_size},
        gradient_accumulation_steps={tc.gradient_accumulation_steps},
        learning_rate={tc.learning_rate},
        lr_scheduler_type="{tc.lr_scheduler_type}",
        warmup_ratio={tc.warmup_ratio}, optim="{tc.optim}",
        weight_decay=0.01, max_grad_norm=1.0, fp16=False, bf16=True,
        logging_steps=1, save_strategy="epoch", seed=42,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={{"use_reentrant": False}},
        report_to="none",
        max_length={tc.max_seq_length},
        dataset_text_field="text",
        completion_only_loss=_use_completion_only,
    ),
)
stats = trainer.train()
print(f"Final loss: {{stats.training_loss:.4f}}")

lora_dir = "{tc.output_dir}/lora_adapters"
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
print("PIPELINE_STAGE_COMPLETE=training")
'''
    rc = await run_script(script, tc.output_dir)
    ok = rc == 0
    await state.set_stage("training", StageStatus.COMPLETE if ok else StageStatus.FAILED)
    if ok:
        await state.set_progress(100)
    return ok


async def do_export(cfg: RunRequest) -> bool:
    """Merge LoRA + export. Smart routing based on upstream/downstream stages."""
    ec = cfg.export
    out = cfg.training.output_dir
    out_abs = _resolve_out(out)
    training_enabled = "training" in cfg.enabled_stages
    mq_enabled = "magicquant" in cfg.enabled_stages

    # Skip if expected artifacts already exist
    if mq_enabled:
        merged = out_abs / "merged_model"
        has_safetensors = merged.exists() and any(merged.glob("*.safetensors"))
        if has_safetensors:
            await state.log(f"Merged safetensors already exist at {merged} — skipping export", "success")
            await state.set_stage("export", StageStatus.COMPLETE)
            await state.set_progress(100)
            return True
    else:
        gguf = out_abs / "model-bf16.gguf"
        if gguf.exists():
            size = gguf.stat().st_size / 1e9
            await state.log(f"GGUF already exists at {gguf} ({size:.1f} GB) — skipping export", "success")
            await state.set_stage("export", StageStatus.COMPLETE)
            await state.set_progress(100)
            return True

    await state.set_stage("export", StageStatus.RUNNING)
    await state.set_progress(0)

    # Determine model source: training output (lora) or user-specified model
    if training_enabled:
        model_source = f"{out}/lora_adapters"
        has_lora = True
    elif ec.source_model:
        model_source = ec.source_model
        # Detect if source is a LoRA adapter dir (has adapter_config.json)
        has_lora = (Path(model_source) / "adapter_config.json").exists()
    else:
        await state.log("Export requires a model source. Enable Training or set a Source Model path.", "error")
        await state.set_stage("export", StageStatus.FAILED)
        return False

    # Validate source exists (for local paths)
    if not model_source.startswith(("http", "hf://")) and "/" in model_source:
        source_path = Path(model_source)
        if source_path.is_absolute() and not source_path.exists():
            await state.log(f"Source model not found: {model_source}", "error")
            await state.set_stage("export", StageStatus.FAILED)
            return False

    load_desc = "LoRA adapters" if has_lora else "model"

    # Shared preamble: env vars + HF cache check
    env_setup = '''
import os
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "backend:native,expandable_segments:True"
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
'''
    cache_check = f'''
from pathlib import Path as _P
_model_id = "{model_source}"
_is_local = _P(_model_id).exists()
if _is_local:
    print(f"Loading from local path: {{_model_id}}")
else:
    try:
        from huggingface_hub import scan_cache_dir
        _cached = False
        for _repo in scan_cache_dir().repos:
            if _repo.repo_id == _model_id:
                _size_gb = _repo.size_on_disk / 1e9
                print(f"Model found in HF cache ({{_size_gb:.1f}} GB) — no download needed")
                _cached = True
                break
        if not _cached:
            print(f"Model not in cache — will download from HuggingFace: {{_model_id}}")
    except Exception:
        print(f"Loading model: {{_model_id}}")
'''

    if mq_enabled:
        if has_lora:
            await state.log(f"Merging {load_desc} to safetensors (MagicQuant reads them directly)", "stage")
            save_line = f'model.save_pretrained_merged("{out}/merged_model", tokenizer, save_method="merged_16bit")'
        else:
            await state.log(f"Saving {load_desc} as safetensors for MagicQuant", "stage")
            save_line = f'''
from peft import PeftModel as _PM
if isinstance(model, _PM):
    model.save_pretrained_merged("{out}/merged_model", tokenizer, save_method="merged_16bit")
else:
    print("Base model (no LoRA) — saving directly with save_pretrained")
    model.save_pretrained("{out}/merged_model")
    tokenizer.save_pretrained("{out}/merged_model")
'''
        script = env_setup + cache_check + f'''
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{model_source}", max_seq_length=4096, load_in_4bit=False,
)
print("Saving safetensors...")
{save_line}
print("PIPELINE_STAGE_COMPLETE=export")
'''
    else:
        await state.log(f"Exporting {load_desc} to {ec.gguf_type.upper()} GGUF", "stage")
        script = env_setup + cache_check + f'''
import shutil
from pathlib import Path
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{model_source}", max_seq_length=4096, load_in_4bit=False,
)

if {ec.also_save_merged}:
    from peft import PeftModel as _PM
    if isinstance(model, _PM):
        print("Also saving merged safetensors...")
        model.save_pretrained_merged("{out}/merged_model", tokenizer, save_method="merged_16bit")
    else:
        print("Base model (no LoRA) — saving safetensors directly...")
        model.save_pretrained("{out}/merged_model")
        tokenizer.save_pretrained("{out}/merged_model")

print("Converting to {ec.gguf_type.upper()} GGUF...")
model.save_pretrained_gguf("{out}", tokenizer, quantization_method="{ec.gguf_type}")

dst = Path("{out}/model-bf16.gguf")
for d in [Path("{out}"), Path("{out}_gguf")]:
    if d.exists():
        for f in d.glob("*.gguf"):
            if f != dst:
                shutil.move(str(f), str(dst))
                break
if dst.exists():
    size = dst.stat().st_size / 1e9
    print(f"GGUF: {{dst}} ({{size:.1f}} GB)")
print("PIPELINE_STAGE_COMPLETE=export")
'''
    rc = await run_script(script, out)
    ok = rc == 0
    await state.set_stage("export", StageStatus.COMPLETE if ok else StageStatus.FAILED)
    if ok:
        await state.set_progress(100)
    return ok


async def do_magicquant(cfg: RunRequest) -> bool:
    """Run MagicQuant evolutionary search and generate tiered hybrid GGUFs."""
    out = cfg.training.output_dir
    out_abs = _resolve_out(out)
    mc = cfg.magicquant
    export_enabled = "export" in cfg.enabled_stages

    # Skip if MagicQuant GGUFs already exist
    mq_dir = out_abs / "magicquant"
    if mq_dir.exists():
        existing_ggufs = list(mq_dir.glob("*.gguf"))
        if existing_ggufs:
            names = ", ".join(f.name for f in existing_ggufs)
            await state.log(f"MagicQuant GGUFs already exist ({names}) — skipping", "success")
            await state.set_stage("magicquant", StageStatus.COMPLETE)
            await state.set_progress(100)
            return True

    await state.set_stage("magicquant", StageStatus.RUNNING)
    await state.set_progress(0)
    await state.log("Starting MagicQuant evolutionary search", "stage")

    tiers_json = json.dumps(mc.tiers)
    # Derive model name from the effective source, not just the training config
    model_name = _derive_model_short_name(cfg)
    hint = mc.llamacpp_path or ""

    # Determine source model for MagicQuant
    if mc.source_model and not export_enabled:
        mq_source_override = mc.source_model
    else:
        mq_source_override = ""

    # Improvement #4: auto-install llama.cpp
    script = f'''
import sys, os, subprocess, multiprocessing
from pathlib import Path

def find_llamacpp():
    for p in ["{hint}", os.environ.get("LLAMACPP_PATH",""),
              str(Path.home()/"llama.cpp"), "./llama.cpp", "/usr/local"]:
        if not p: continue
        pp = Path(p)
        for sub in [pp/"convert_hf_to_gguf.py", pp/"build"/"bin"/"llama-perplexity"]:
            if sub.exists(): return str(pp)
    return None

llamacpp = find_llamacpp()
if not llamacpp:
    install_dir = Path.home() / "llama.cpp"
    print("llama.cpp not found — auto-installing...")
    rc = subprocess.run(["git", "clone", "--depth", "1",
                         "https://github.com/ggml-org/llama.cpp.git", str(install_dir)]).returncode
    if rc == 0:
        build_dir = install_dir / "build"
        rc = subprocess.run(["cmake", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
                             str(install_dir)]).returncode
        if rc == 0:
            jobs = str(multiprocessing.cpu_count())
            rc = subprocess.run(["cmake", "--build", str(build_dir), "-j", jobs]).returncode
    if rc == 0:
        llamacpp = str(install_dir)
        print(f"llama.cpp installed: {{llamacpp}}")
    else:
        print("Warning: llama.cpp install failed, using heuristic probing")
        llamacpp = None

print(f"llama.cpp: {{llamacpp or 'not found (heuristic mode)'}}")

from magicquant.orchestrator import MagicQuantOrchestrator
import json

# Determine source: explicit override > merged safetensors > GGUF
override = "{mq_source_override}"
if override:
    source = override
elif Path("{out}/merged_model").exists():
    source = "{out}/merged_model"
elif Path("{out}/model-bf16.gguf").exists():
    source = "{out}/model-bf16.gguf"
else:
    print("Error: no source model found. Enable Export or set a Source Model path in MagicQuant config.")
    sys.exit(1)
print(f"MagicQuant source: {{source}}")

orch = MagicQuantOrchestrator(
    source_model_path=source,
    output_dir="{out}/magicquant",
    llamacpp_path=llamacpp,
)

print(f"Search: generations={mc.generations}, population={mc.population_size}, base={mc.target_base_quant}")

best_configs, tiered = orch.run_full_search(
    target_base_quant="{mc.target_base_quant}",
    max_generations={mc.generations},
    population_size={mc.population_size},
    verbose=True,
)

if not tiered:
    print("Error: no viable configurations found")
    sys.exit(1)

print(f"Tiers found: {{list(tiered.keys())}}")

tiers = json.loads('{tiers_json}')
paths = orch.generate_tiered_models(
    tiered=tiered, model_name_prefix="{model_name}", tiers=tiers, verify=False,
)
valid = [p for p in paths if p]
for p in valid:
    size = os.path.getsize(p) / 1e9
    print(f"  {{Path(p).name}} ({{size:.1f}} GB)")
print(f"Generated {{len(valid)}} hybrid GGUF files")
print("PIPELINE_STAGE_COMPLETE=magicquant")
'''
    rc = await run_script(script, out)
    ok = rc == 0
    await state.set_stage("magicquant", StageStatus.COMPLETE if ok else StageStatus.FAILED)
    if ok:
        await state.set_progress(100)
    return ok


async def do_upload(cfg: RunRequest) -> bool:
    """Upload pipeline artifacts (GGUF, LoRA, merged) to HuggingFace Hub."""
    out = cfg.training.output_dir
    uc = cfg.upload
    await state.set_stage("upload", StageStatus.RUNNING)
    await state.set_progress(0)
    await state.log("Uploading to HuggingFace Hub", "stage")

    if not uc or not uc.repo_id:
        await state.log("No repo_id configured", "error")
        await state.set_stage("upload", StageStatus.FAILED)
        return False

    base_model = cfg.training.model_name
    lora_r = cfg.training.lora_r
    lora_alpha = cfg.training.lora_alpha

    script = f'''
import os, sys
from pathlib import Path
from huggingface_hub import HfApi, ModelCard, ModelCardData

api = HfApi()
repo_id = "{uc.repo_id}"
out = "{out}"

print(f"Creating repo: {{repo_id}}")
api.create_repo(repo_id=repo_id, repo_type="model", private={uc.private}, exist_ok=True)

files = []

if {uc.upload_lora}:
    lora_dir = Path(out) / "lora_adapters"
    if lora_dir.exists():
        for f in lora_dir.iterdir():
            if f.is_file(): files.append((str(f), f"lora/{{f.name}}"))

if {uc.upload_merged}:
    merged_dir = Path(out) / "merged_model"
    if merged_dir.exists():
        for f in merged_dir.iterdir():
            if f.is_file(): files.append((str(f), f"merged/{{f.name}}"))

if {uc.upload_gguf}:
    gguf_files = []
    mq_dir = Path(out) / "magicquant"
    if mq_dir.exists():
        gguf_files = list(mq_dir.glob("*.gguf"))
    if not gguf_files:
        bf16 = Path(out) / "model-bf16.gguf"
        if bf16.exists(): gguf_files = [bf16]
    for f in gguf_files:
        files.append((str(f), f.name))

if not files:
    print("Error: no files to upload")
    sys.exit(1)

print(f"Uploading {{len(files)}} files...")

gguf_table = ""
for local, name in files:
    if name.endswith(".gguf"):
        size = os.path.getsize(local) / 1e9
        gguf_table += f"| {{name}} | {{size:.1f}} GB |\\n"

card_data = ModelCardData(
    license="{uc.license}", library_name="llama.cpp",
    base_model="{base_model}", pipeline_tag="text-generation",
    tags=["gguf", "quantized", "magicquant"],
)
card_md = f"""---
{{card_data.to_yaml()}}
---

# {{repo_id.split("/")[-1]}}

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}).

| File | Size |
|------|------|
{{gguf_table}}

*Generated with the MagicQuant Pipeline (QLoRA r={lora_r}, alpha={lora_alpha}, completion-only loss)*
"""
try:
    ModelCard(card_md).push_to_hub(repo_id)
    print("Model card uploaded")
except Exception as e:
    print(f"Warning: model card failed: {{e}}")

for i, (local, name) in enumerate(files, 1):
    size = os.path.getsize(local) / 1e9
    print(f"[{{i}}/{{len(files)}}] {{name}} ({{size:.1f}} GB)...")
    api.upload_file(path_or_fileobj=local, path_in_repo=name,
                    repo_id=repo_id, repo_type="model", commit_message=f"Add {{name}}")
    print(f"  Done")

print(f"https://huggingface.co/{{repo_id}}")
print("PIPELINE_STAGE_COMPLETE=upload")
'''
    rc = await run_script(script, out)
    ok = rc == 0
    await state.set_stage("upload", StageStatus.COMPLETE if ok else StageStatus.FAILED)
    if ok:
        await state.set_progress(100)
    return ok


# ── Pipeline orchestration ───────────────────────────────────────────────────

STAGE_RUNNERS = {
    "training":   do_training,
    "export":     do_export,
    "magicquant": do_magicquant,
    "upload":     do_upload,
}

async def validate_pipeline(cfg: RunRequest) -> bool:
    """Pre-flight checks for stage dependencies."""
    enabled = set(cfg.enabled_stages)
    out_abs = _resolve_out(cfg.training.output_dir)

    # Export without training: needs a source model
    if "export" in enabled and "training" not in enabled:
        source = cfg.export.source_model if cfg.export else ""
        if not source:
            await state.log("Export is enabled without Training, but no Source Model is set. "
                            "Provide a HuggingFace model ID or local path in the Export config.", "error")
            return False
        await state.log(f"Training skipped — Export will use source model: {source}")

    # MagicQuant without export: needs a source
    if "magicquant" in enabled and "export" not in enabled:
        mc = cfg.magicquant
        source = mc.source_model if mc else ""
        # Check if prior pipeline output exists
        has_merged = (out_abs / "merged_model").exists()
        has_gguf = (out_abs / "model-bf16.gguf").exists()
        if not source and not has_merged and not has_gguf:
            await state.log("MagicQuant is enabled without Export, and no existing model artifacts "
                            "were found in the output directory. Set a Source Model path in MagicQuant "
                            "config, or enable Export.", "error")
            return False
        if source:
            await state.log(f"Export skipped — MagicQuant will use source: {source}")
        elif has_merged:
            await state.log(f"Export skipped — MagicQuant will use existing: {out_abs}/merged_model")
        else:
            await state.log(f"Export skipped — MagicQuant will use existing: {out_abs}/model-bf16.gguf")

    # Upload: check that at least some artifacts will exist
    if "upload" in enabled:
        uc = cfg.upload
        if not uc or not uc.repo_id:
            await state.log("Upload is enabled but Repository ID is empty.", "error")
            return False

    return True


def _derive_model_short_name(cfg: RunRequest) -> str:
    """Extract a short model name from the first available source across stages."""
    enabled = set(cfg.enabled_stages)
    if "training" in enabled:
        raw = cfg.training.model_name
    elif "export" in enabled and cfg.export and cfg.export.source_model:
        raw = cfg.export.source_model
    elif "magicquant" in enabled and cfg.magicquant and cfg.magicquant.source_model:
        raw = cfg.magicquant.source_model
    else:
        raw = cfg.training.model_name
    # Strip org/user prefix, path components, and known model file extensions
    name = raw.rstrip("/").split("/")[-1]
    for ext in (".gguf", ".safetensors", ".bin", ".pt", ".pth"):
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    # Sanitize for filesystem
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name).strip("-") or "model"


async def run_pipeline(cfg: RunRequest):
    """Execute enabled pipeline stages in order: training, export, magicquant, upload."""
    state.running = True
    enabled = set(cfg.enabled_stages)

    # Create model-specific output subdirectory
    model_name = _derive_model_short_name(cfg)
    base_out = cfg.training.output_dir
    cfg.training.output_dir = f"{base_out}/{model_name}"
    out_abs = _resolve_out(cfg.training.output_dir)
    out_abs.mkdir(parents=True, exist_ok=True)
    await state.log(f"Output directory: {out_abs}", "info")

    for s in ALL_STAGES:
        await state.set_stage(s, StageStatus.SKIPPED if s not in enabled else StageStatus.PENDING)

    try:
        if not await validate_pipeline(cfg):
            await state.log("Pipeline aborted due to validation errors.", "error")
            return

        for stage_name in ALL_STAGES:
            if stage_name not in enabled:
                continue
            ok = await STAGE_RUNNERS[stage_name](cfg)
            if not ok:
                await state.log(f"Pipeline stopped at {stage_name}", "error")
                break

        if all(state.stages[s] in (StageStatus.COMPLETE, StageStatus.SKIPPED) for s in ALL_STAGES):
            await state.log("Pipeline complete!", "success")
    except Exception as e:
        await state.log(f"Pipeline error: {e}", "error")
    finally:
        state.running = False
        await state.broadcast({"type": "pipeline_done"})


# ── Persistent config ────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.json"

def load_config() -> dict:
    """Load persisted UI config from config.json, or return empty dict on failure."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}

def save_config(cfg: dict):
    """Persist UI config to config.json."""
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page frontend HTML."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/state")
async def get_state():
    """Return the current pipeline state: stage statuses, running flag, and progress."""
    return {
        "stages": {k: v.value for k, v in state.stages.items()},
        "running": state.running,
        "current_stage": state.current_stage,
        "progress": state.progress,
    }


@app.get("/api/config")
async def get_config():
    """Return the persisted UI configuration (e.g. HuggingFace username)."""
    return load_config()


@app.post("/api/config")
async def set_config(body: dict):
    """Merge and persist UI configuration values. Returns the updated config."""
    cfg = load_config()
    cfg.update(body)
    save_config(cfg)
    return cfg


@app.post("/api/run")
async def start_pipeline(cfg: RunRequest):
    """
    Launch the pipeline in a background task.

    Accepts a full RunRequest with per-stage config and an enabled_stages list.
    Returns an error if a pipeline is already in progress.
    """
    if state.running:
        return {"error": "Pipeline is already running"}
    for s in ALL_STAGES:
        state.stages[s] = StageStatus.PENDING
    state.progress = 0
    asyncio.create_task(run_pipeline(cfg))
    return {"status": "started"}


@app.post("/api/stop")
async def stop_pipeline():
    """Request a graceful pipeline stop. Sets the running flag to False."""
    if not state.running:
        return {"error": "Pipeline is not running"}
    state.running = False
    await state.log("Stop requested by user", "warn")
    await state.broadcast({"type": "pipeline_done"})
    return {"status": "stopping"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for real-time log streaming.

    Clients connect here to receive log lines, stage updates, and progress
    events as JSON messages. The connection stays open until the client
    disconnects.
    """
    await ws.accept()
    state.ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PIPELINE_UI_PORT", 7865))
    uvicorn.run(app, host="0.0.0.0", port=port)
