#!/usr/bin/env python3
"""
Foundry UI — FastAPI backend.

Orchestrates a 4-stage LLM fine-tuning pipeline:
  Training → Export GGUF → MagicQuant → Upload

Uses WebSocket for real-time log streaming to the browser.
Port defaults to 7865 (configurable via FOUNDRY_UI_PORT env var).
"""

import asyncio
import json
import os
import re
import signal
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Header, Query
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from config import settings as foundry_settings
from services import (
    TrainingService,
    ExportService,
    HereticService,
    ReapService,
    MagicQuantService,
    UploadService,
)

API_KEY = os.environ.get("FOUNDRY_API_KEY", "")

async def verify_api_key(authorization: str = Header(default="")):
    """Check Bearer token in the Authorization header. No-op when API_KEY is unset."""
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")

app = FastAPI(title="Foundry")

FOUNDRY_DIR = Path(__file__).resolve().parent.parent


def _resolve_venv_python() -> str:
    """Locate the venv Python interpreter at runtime."""
    candidate = FOUNDRY_DIR / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


VENV_PYTHON = _resolve_venv_python()


# ── State ────────────────────────────────────────────────────────────────────

class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"

ALL_STAGES = ["training", "export", "heretic", "reap", "magicquant", "upload"]

class PipelineState:
    """Shared mutable state for the running pipeline, including WebSocket fan-out."""

    def __init__(self):
        self.stages = {s: StageStatus.PENDING for s in ALL_STAGES}
        self.running = False
        self.current_stage = None
        self.progress = 0
        self.ws_clients: list[WebSocket] = []
        self.active_proc = None

    async def broadcast(self, msg: dict):
        dead = []
        for ws in list(self.ws_clients):
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
    datasets: list[str] = ["data/zeroclaw_training_data.jsonl"]
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
    warmup_steps: int = 10
    optim: str = "paged_adamw_8bit"
    packing: bool = False
    output_dir: str = "./output"

class ExportCfg(BaseModel):
    gguf_type: str = "bf16"
    also_save_merged: bool = False
    source_model: str = ""  # when training is skipped: HF ID or local path to model/lora

class HereticCfg(BaseModel):
    n_trials: int = 200
    n_startup_trials: int = 60
    quantization: str = "bnb_4bit"
    kl_divergence_scale: float = 1.0
    orthogonalize_direction: bool = False
    row_normalization: str = "none"

class ReapCfg(BaseModel):
    compression_ratio: float = 0.25
    prune_method: str = "reap"
    samples_per_category: int = 512
    model_max_length: int = 2048
    dataset_name: str = "theblackcat102/evol-codealpaca-v1"
    seed: int = 42

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
    license: str = ""  # empty = auto-detect from base model
    upload_gguf: bool = True
    upload_lora: bool = False
    upload_merged: bool = False
    upload_dataset: bool = True

class RunRequest(BaseModel):
    training: TrainingCfg = TrainingCfg()
    export: Optional[ExportCfg] = ExportCfg()
    heretic: Optional[HereticCfg] = None
    reap: Optional[ReapCfg] = None
    magicquant: Optional[MagicQuantCfg] = None
    upload: Optional[UploadCfg] = None
    enabled_stages: list[str] = ["training", "export"]


# ── Subprocess helper ────────────────────────────────────────────────────────

async def run_script(script: str, output_dir: str) -> int:
    """Write a Python script to disk and execute it in the venv, streaming stdout to WebSocket clients."""
    # Resolve relative paths against the project root, not uvicorn's CWD
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = FOUNDRY_DIR / out_path
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
    # Auto-load HF token from cache if not in environment
    if "HF_TOKEN" not in env:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            env["HF_TOKEN"] = token_path.read_text().strip()

    log_path = script_path.with_suffix(".log")
    log_file = open(log_path, "w")

    try:
        proc = await asyncio.create_subprocess_exec(
            VENV_PYTHON, "-u", str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env, cwd=str(FOUNDRY_DIR),
            limit=1024 * 1024,  # 1MB line buffer — tqdm \r bars can be huge
            start_new_session=True,
        )
        state.active_proc = proc

        try:
            async for raw in proc.stdout:
                log_file.write(raw.decode("utf-8", errors="replace"))
                log_file.flush()
                # tqdm uses \r without \n, so one "line" may contain many \r-separated updates.
                # Split on \r and process the last (most recent) segment.
                segments = raw.decode("utf-8", errors="replace").split("\r")
                for text in segments:
                    text = text.strip()
                    if not text:
                        continue
                    # Parse progress from tqdm bars and shard-loading output.
                    # Two phases: loading = 0-15%, training/processing = 15-100%.
                    # Shard loading: "Done in 2.1s | Progress: 45/1275 (4%)"
                    # Tqdm training: "  5%|█ | 3/57 [00:15<04:30, 5.00s/it]"
                    pct_parsed = False
                    if "Progress:" in text and "%" in text:
                        try:
                            raw_pct = int(text.split("(")[1].split("%")[0])
                            await state.set_progress(int(raw_pct * 0.15))
                            pct_parsed = True
                        except (ValueError, IndexError):
                            pass
                    elif "%|" in text:
                        try:
                            raw_pct = int(float(text.split("%|")[0].strip().split()[-1]))
                            if "Fetching" in text or "Map:" in text or "Tokenizing" in text:
                                pass  # ignore quick preprocessing tqdm bars
                            else:
                                # Training/export tqdm: map 0-100% to 15-100%
                                await state.set_progress(15 + int(raw_pct * 0.85))
                            pct_parsed = True
                        except (ValueError, IndexError):
                            pass
                    if pct_parsed:
                        continue  # don't log every progress bar update
                    if "'loss'" in text:
                        await state.log(text, "metric")
                    elif "PIPELINE_STAGE_COMPLETE" in text:
                        await state.log(text, "success")
                    elif "Error" in text or "error" in text.lower():
                        await state.log(text, "error")
                    else:
                        await state.log(text)
        except Exception:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            raise
        finally:
            await proc.wait()
            state.active_proc = None
    finally:
        log_file.close()

    return proc.returncode


# ── Dataset validation (Improvement #3) ──────────────────────────────────────

FOUNDRY_ROOT = Path(__file__).resolve().parent.parent


async def validate_dataset(sources: list[str]) -> bool:
    """Pre-flight dataset check for one or more dataset sources."""
    await state.log("Validating dataset(s)...", "stage")

    if not sources or all(not s.strip() for s in sources):
        await state.log("No datasets configured", "error")
        return False

    all_ok = True
    for src in sources:
        src = src.strip()
        if not src:
            continue

        # Strip config/split suffixes for path detection
        clean = src
        if "[" in clean and clean.endswith("]"):
            clean = clean.split("[")[0]
        if ":" in clean and not clean.startswith("/") and not Path(clean).suffix:
            clean = clean.rsplit(":", 1)[0]

        local = Path(clean)
        if local.suffix in (".jsonl", ".json", ".csv", ".parquet"):
            # Local file -- full validation
            p = local
            if not p.is_absolute():
                p = FOUNDRY_ROOT / p
            if not p.exists():
                await state.log(f"Dataset not found: {src}", "error")
                all_ok = False
                continue
            if p.stat().st_size == 0:
                await state.log(f"Dataset file is empty: {src}", "error")
                all_ok = False
                continue

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
                        errors.append(f"Line {i}: invalid JSON -- {e}")
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
                await state.log(f"Validation failed for {src} ({len(errors)} errors)", "error")
                all_ok = False
                continue

            if n < 10:
                await state.log(f"  Warning: only {n} examples in {src}", "warn")
            await state.log(f"  {src}: {n} examples, {tool_calls} tool-call turns, roles: {sorted(roles)}")
        else:
            # Possibly a HuggingFace dataset -- check if it exists locally first
            if local.exists():
                await state.log(f"  Local path: {src}")
            else:
                await state.log(f"  HF dataset: {src} (will be downloaded if not cached)")

    if all_ok:
        await state.log("Dataset validation passed", "success")
    return all_ok


# ── Stage runners ────────────────────────────────────────────────────────────

def _resolve_out(output_dir: str) -> Path:
    """Resolve output_dir the same way run_script does."""
    p = Path(output_dir)
    return p if p.is_absolute() else FOUNDRY_DIR / p


async def do_training(cfg: RunRequest) -> bool:
    """Run the QLoRA training stage. Skips if LoRA adapters already exist."""
    tc = cfg.training
    out = _resolve_out(tc.output_dir)

    # Skip if LoRA adapters already exist.
    # NOTE: This only checks for artifact existence, not that they match the
    # current run config (same model, same dataset, same hyperparameters).
    # A previous run's output for a different config would cause a false skip.
    # A full fix would hash the config and store it alongside the artifacts.
    adapter_cfg = out / "lora_adapters" / "adapter_config.json"
    if adapter_cfg.exists():
        await state.log(f"LoRA adapters already exist at {out / 'lora_adapters'} — skipping training", "success")
        await state.set_stage("training", StageStatus.COMPLETE)
        await state.set_progress(100)
        return True

    # Validate dataset(s) before committing GPU time
    if not await validate_dataset(tc.datasets):
        return False

    await state.set_stage("training", StageStatus.RUNNING)
    await state.set_progress(0)
    await state.log("Starting QLoRA training (completion-only loss)", "stage")

    svc = TrainingService(FOUNDRY_ROOT, VENV_PYTHON)
    script = svc.build_script(
        model_name=tc.model_name,
        datasets=tc.datasets,
        output_dir=tc.output_dir,
        max_seq_length=tc.max_seq_length,
        lora_r=tc.lora_r,
        lora_alpha=tc.lora_alpha,
        lora_dropout=tc.lora_dropout,
        use_rslora=tc.use_rslora,
        num_train_epochs=tc.num_train_epochs,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        lr_scheduler_type=tc.lr_scheduler_type,
        warmup_steps=tc.warmup_steps,
        optim=tc.optim,
        packing=tc.packing,
    )
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

    # Skip if expected artifacts already exist.
    # NOTE: This only checks for artifact existence, not that they match the
    # current run config. Artifacts from a different model/config would cause
    # a false skip. A full fix would hash the config and compare.
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

    # Determine model source: base model ID for streaming_merge + optional LoRA dir
    if training_enabled:
        base_model_id = cfg.training.model_name
        lora_source = f"{out}/lora_adapters"
        has_lora = True
    elif ec.source_model:
        base_model_id = ec.source_model
        has_lora = (Path(ec.source_model) / "adapter_config.json").exists()
        if has_lora:
            try:
                adapter_cfg_data = json.loads((Path(ec.source_model) / "adapter_config.json").read_text())
                base_model_id = adapter_cfg_data.get("base_model_name_or_path", ec.source_model)
            except (json.JSONDecodeError, OSError):
                pass
            lora_source = ec.source_model
        else:
            lora_source = None
    else:
        await state.log("Export requires a model source. Enable Training or set a Source Model path.", "error")
        await state.set_stage("export", StageStatus.FAILED)
        return False

    # Validate source exists (for local paths)
    if not base_model_id.startswith(("http", "hf://")) and "/" in base_model_id:
        source_path = Path(base_model_id)
        if source_path.is_absolute() and not source_path.exists():
            await state.log(f"Source model not found: {base_model_id}", "error")
            await state.set_stage("export", StageStatus.FAILED)
            return False

    load_desc = "LoRA adapters" if has_lora else "model"
    if mq_enabled:
        desc = f"Merging {load_desc} to safetensors" if has_lora else f"Saving {load_desc} as safetensors for MagicQuant"
    else:
        desc = f"Exporting {load_desc} to safetensors (for GGUF conversion)"
    await state.log(desc, "stage")

    svc = ExportService(FOUNDRY_ROOT, VENV_PYTHON)
    script = svc.build_script(
        base_model_id=base_model_id,
        lora_source=lora_source,
        has_lora=has_lora,
        merged_dir=str(out_abs / "merged_model"),
    )
    rc = await run_script(script, out)
    ok = rc == 0
    await state.set_stage("export", StageStatus.COMPLETE if ok else StageStatus.FAILED)
    if ok:
        await state.set_progress(100)
    return ok


async def do_heretic(cfg: RunRequest) -> bool:
    """Run heretic abliteration on the merged model to remove safety alignment."""
    out = cfg.training.output_dir
    out_abs = _resolve_out(out)
    hc = cfg.heretic

    # Skip if heretic output already exists.
    # NOTE: No config-match check -- stale artifacts from a different run
    # would cause a false skip. See do_training() for the same limitation.
    heretic_dir = out_abs / "heretic_model"
    if heretic_dir.exists() and any(heretic_dir.glob("*.safetensors")):
        await state.log(f"Abliterated model already exists at {heretic_dir} -- skipping heretic", "success")
        await state.set_stage("heretic", StageStatus.COMPLETE)
        await state.set_progress(100)
        return True

    # Determine model source: prefer merged_model from export stage
    merged_dir = out_abs / "merged_model"
    if not merged_dir.exists() or not any(merged_dir.glob("*.safetensors")):
        await state.log("No merged model found -- run export first", "error")
        await state.set_stage("heretic", StageStatus.FAILED)
        return False

    await state.set_stage("heretic", StageStatus.RUNNING)
    await state.set_progress(0)
    await state.log("Starting heretic abliteration (Optuna-optimized directional ablation)", "stage")

    checkpoint_dir = str(out_abs / "_heretic_checkpoints")

    svc = HereticService(FOUNDRY_ROOT, VENV_PYTHON)
    script = svc.build_script(
        model_path=str(merged_dir),
        output_path=str(heretic_dir),
        checkpoint_dir=checkpoint_dir,
        n_trials=hc.n_trials,
        n_startup_trials=hc.n_startup_trials,
        quantization=hc.quantization,
        kl_divergence_scale=hc.kl_divergence_scale,
        orthogonalize_direction=hc.orthogonalize_direction,
        row_normalization=hc.row_normalization,
    )
    rc = await run_script(script, out)
    ok = rc == 0
    await state.set_stage("heretic", StageStatus.COMPLETE if ok else StageStatus.FAILED)
    if ok:
        await state.set_progress(100)
    return ok


REAP_SUPPORTED_ARCHS = {
    "Qwen3MoeForCausalLM",
    "Qwen3-Coder-30B-A3B-Instruct",
    "NonUniformQwen3MoeForCausalLM",
    "Llama4ForCausalLM",
    "MixtralForCausalLM",
    "DeepseekV2ForCausalLM",
    "Ernie4_5_MoEForCausalLM",
    "Ernie4_5_MoeForCausalLM",
    "gpt-oss-20b",
    "Glm4MoeForCausalLM",
}


def _detect_model_arch(model_path: Path) -> Optional[str]:
    """Read config.json and return the first entry in architectures list, or None."""
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        return None
    try:
        data = json.loads(cfg_path.read_text())
        archs = data.get("architectures") or []
        if archs and isinstance(archs, list):
            return archs[0]
    except (json.JSONDecodeError, OSError):
        pass
    return None


async def do_reap(cfg: RunRequest) -> bool:
    """Run REAP expert pruning on the merged or abliterated model.

    Reads from heretic_model/ (preferred) or merged_model/ and writes to
    reap_model/. Silently skips (returns True) if the architecture is not
    supported by REAP.
    """
    out = cfg.training.output_dir
    out_abs = _resolve_out(out)
    rc = cfg.reap

    # Skip if REAP output already exists.
    reap_dir = out_abs / "reap_model"
    if reap_dir.exists() and any(reap_dir.glob("*.safetensors")):
        await state.log(f"Pruned model already exists at {reap_dir} — skipping REAP", "success")
        await state.set_stage("reap", StageStatus.COMPLETE)
        await state.set_progress(100)
        return True

    # Determine model source: prefer heretic output, fall back to merged_model
    heretic_dir = out_abs / "heretic_model"
    merged_dir = out_abs / "merged_model"
    if heretic_dir.exists() and any(heretic_dir.glob("*.safetensors")):
        source_dir = heretic_dir
        await state.log(f"REAP source: abliterated model at {source_dir}")
    elif merged_dir.exists() and any(merged_dir.glob("*.safetensors")):
        source_dir = merged_dir
        await state.log(f"REAP source: merged model at {source_dir}")
    else:
        await state.log("No merged or abliterated model found -- run export first", "error")
        await state.set_stage("reap", StageStatus.FAILED)
        return False

    # Check architecture support before launching a subprocess.
    arch = _detect_model_arch(source_dir)
    if arch is None:
        await state.log(
            f"Could not detect model architecture from {source_dir}/config.json — skipping REAP",
            "warn",
        )
        await state.set_stage("reap", StageStatus.SKIPPED)
        return True
    if arch not in REAP_SUPPORTED_ARCHS:
        await state.log(
            f"Architecture '{arch}' is not supported by REAP — skipping stage",
            "warn",
        )
        await state.log(f"  REAP supports: {sorted(REAP_SUPPORTED_ARCHS)}")
        await state.set_stage("reap", StageStatus.SKIPPED)
        return True
    await state.log(f"Detected supported REAP architecture: {arch}")

    await state.set_stage("reap", StageStatus.RUNNING)
    await state.set_progress(0)
    await state.log("Starting REAP expert pruning", "stage")

    svc = ReapService(FOUNDRY_ROOT, VENV_PYTHON)
    script = svc.build_script(
        input_dir=str(source_dir.resolve()),
        output_dir=str(reap_dir.resolve()),
        cwd_dir=str(out_abs.resolve()),
        compression_ratio=rc.compression_ratio,
        prune_method=rc.prune_method,
        samples_per_category=rc.samples_per_category,
        model_max_length=rc.model_max_length,
        dataset_name=rc.dataset_name,
        seed=rc.seed,
    )
    rc_code = await run_script(script, out)
    ok = rc_code == 0
    await state.set_stage("reap", StageStatus.COMPLETE if ok else StageStatus.FAILED)
    if ok:
        await state.set_progress(100)
    return ok


async def do_magicquant(cfg: RunRequest) -> bool:
    """Run MagicQuant evolutionary search and generate tiered hybrid GGUFs."""
    out = cfg.training.output_dir
    out_abs = _resolve_out(out)
    mc = cfg.magicquant
    export_enabled = "export" in cfg.enabled_stages

    # Skip if MagicQuant GGUFs already exist.
    # NOTE: No config-match check -- stale artifacts from a different run
    # would cause a false skip. See do_training() for the same limitation.
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
    model_name = _derive_model_short_name(cfg)
    hint = mc.llamacpp_path or ""
    mq_source_override = mc.source_model if (mc.source_model and not export_enabled) else ""

    svc = MagicQuantService(FOUNDRY_ROOT, VENV_PYTHON)
    script = svc.build_script(
        llamacpp_hint=hint,
        pipeline_root_str=str(FOUNDRY_ROOT),
        mq_source_override=mq_source_override,
        out_abs_str=str(out_abs),
        generations=mc.generations,
        population_size=mc.population_size,
        target_base_quant=mc.target_base_quant,
        tiers_json=tiers_json,
        model_name=model_name,
    )
    rc = await run_script(script, out)
    ok = rc == 0
    await state.set_stage("magicquant", StageStatus.COMPLETE if ok else StageStatus.FAILED)
    if ok:
        await state.set_progress(100)
    return ok


async def do_upload(cfg: RunRequest) -> bool:
    """Upload pipeline artifacts (GGUF, LoRA, merged) to HuggingFace Hub.

    Delegates to hf_upload module for model card generation, progress reporting,
    and file upload.
    """
    out = cfg.training.output_dir
    uc = cfg.upload
    await state.set_stage("upload", StageStatus.RUNNING)
    await state.set_progress(0)
    await state.log("Uploading to HuggingFace Hub", "stage")

    if not uc or not uc.repo_id:
        await state.log("No repo_id configured", "error")
        await state.set_stage("upload", StageStatus.FAILED)
        return False

    tc = cfg.training
    out_abs = _resolve_out(out)

    enabled = set(cfg.enabled_stages)
    # Auto-detect license from base model if not explicitly set
    license_id = uc.license
    if not license_id or license_id == "auto":
        await state.log("Detecting license from base model...")
        try:
            from huggingface_hub import model_info as _mi
            _info = _mi(tc.model_name)
            for _tag in (_info.tags or []):
                if _tag.startswith("license:"):
                    license_id = _tag.split(":", 1)[1]
                    break
            if not license_id or license_id == "auto":
                license_id = getattr(getattr(_info, 'card_data', None), 'license', '') or "unknown"
            await state.log(f"  License: {license_id}")
        except Exception as e:
            license_id = "unknown"
            await state.log(f"  Could not detect license: {e}", "warn")

    svc = UploadService(FOUNDRY_ROOT, VENV_PYTHON)
    script = svc.build_script(
        repo_id=uc.repo_id,
        private=uc.private,
        license_id=license_id,
        upload_gguf=uc.upload_gguf,
        upload_lora=uc.upload_lora,
        upload_merged=uc.upload_merged,
        upload_dataset=uc.upload_dataset,
        base_model=tc.model_name,
        dataset_name=tc.datasets[0] if tc.datasets else "",
        did_training="training" in enabled,
        did_heretic="heretic" in enabled,
        did_reap="reap" in enabled,
        did_magicquant="magicquant" in enabled,
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
        out_abs=str(out_abs),
    )
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
    "heretic":    do_heretic,
    "reap":       do_reap,
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

    # Heretic without export: needs merged model from prior run
    if "heretic" in enabled and "export" not in enabled:
        has_merged = (out_abs / "merged_model").exists()
        if not has_merged:
            await state.log("Heretic is enabled without Export, and no merged model was found "
                            "in the output directory. Enable Export to produce a merged model first.", "error")
            return False
        await state.log(f"Export skipped -- Heretic will use existing: {out_abs}/merged_model")

    # REAP without export/heretic: needs a merged or abliterated model from prior run
    if "reap" in enabled and "export" not in enabled and "heretic" not in enabled:
        has_heretic = (out_abs / "heretic_model").exists()
        has_merged = (out_abs / "merged_model").exists()
        if not has_heretic and not has_merged:
            await state.log("REAP is enabled without Export or Heretic, and no merged or abliterated "
                            "model was found in the output directory. Enable an upstream stage or run "
                            "a pipeline to produce a safetensors model first.", "error")
            return False
        target = "heretic_model" if has_heretic else "merged_model"
        await state.log(f"Upstream stages skipped — REAP will use existing: {out_abs}/{target}")

    # MagicQuant without export: needs a source
    if "magicquant" in enabled and "export" not in enabled:
        mc = cfg.magicquant
        source = mc.source_model if mc else ""
        # Check if prior pipeline output exists
        has_reap = (out_abs / "reap_model").exists()
        has_heretic = (out_abs / "heretic_model").exists()
        has_merged = (out_abs / "merged_model").exists()
        has_gguf = (out_abs / "model-bf16.gguf").exists()
        if not source and not has_reap and not has_heretic and not has_merged and not has_gguf:
            await state.log("MagicQuant is enabled without Export, and no existing model artifacts "
                            "were found in the output directory. Set a Source Model path in MagicQuant "
                            "config, or enable Export.", "error")
            return False
        if source:
            await state.log(f"Export skipped — MagicQuant will use source: {source}")
        elif has_reap:
            await state.log(f"Export skipped — MagicQuant will use existing: {out_abs}/reap_model")
        elif has_heretic:
            await state.log(f"Export skipped — MagicQuant will use existing: {out_abs}/heretic_model")
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
    state.current_stage = None
    state.progress = 0
    enabled = set(cfg.enabled_stages)

    # Create model-specific output subdirectory (avoid nested dirs on re-run)
    model_name = _derive_model_short_name(cfg)
    base_out = cfg.training.output_dir
    if not base_out.rstrip("/").endswith(f"/{model_name}") and Path(base_out).name != model_name:
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
            if not state.running:
                await state.log("Pipeline stopped by user", "warn")
                break
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

@app.get("/health")
async def health_check():
    """Health check endpoint -- no authentication required."""
    return {"status": "ok", "auth_enabled": bool(API_KEY)}


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page frontend HTML. No auth required -- the JS frontend handles auth."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/state", dependencies=[Depends(verify_api_key)])
async def get_state():
    """Return the current pipeline state: stage statuses, running flag, and progress."""
    return {
        "stages": {k: v.value for k, v in state.stages.items()},
        "running": state.running,
        "current_stage": state.current_stage,
        "progress": state.progress,
    }


@app.get("/api/config", dependencies=[Depends(verify_api_key)])
async def get_config():
    """Return the persisted UI configuration (e.g. HuggingFace username)."""
    return load_config()


@app.post("/api/config", dependencies=[Depends(verify_api_key)])
async def set_config(body: dict):
    """Merge and persist UI configuration values. Returns the updated config."""
    cfg = load_config()
    cfg.update(body)
    save_config(cfg)
    return cfg


# ── Workflow save/restore ───────────────────────────────────────────────────

WORKFLOW_DIR = Path.home() / ".foundry" / "workflows"
_WORKFLOW_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


@app.get("/api/workflows", dependencies=[Depends(verify_api_key)])
async def list_workflows():
    """List saved workflow names."""
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(WORKFLOW_DIR.glob("*.json"))
    return {"workflows": [f.stem for f in files]}


@app.post("/api/workflows/{name}", dependencies=[Depends(verify_api_key)])
async def save_workflow(name: str, cfg: RunRequest):
    """Save current config as a named workflow."""
    if not _WORKFLOW_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid workflow name. Use alphanumeric, hyphens, and underscores only.")
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "name": name,
        "saved_at": time.time(),
        "enabled_stages": cfg.enabled_stages,
        "config": cfg.model_dump(),
    }
    path = WORKFLOW_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    return {"status": "saved", "name": name}


@app.get("/api/workflows/{name}", dependencies=[Depends(verify_api_key)])
async def load_workflow(name: str):
    """Load a saved workflow."""
    if not _WORKFLOW_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid workflow name.")
    path = WORKFLOW_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
    data = json.loads(path.read_text())
    return data


@app.delete("/api/workflows/{name}", dependencies=[Depends(verify_api_key)])
async def delete_workflow(name: str):
    """Delete a saved workflow."""
    if not _WORKFLOW_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid workflow name.")
    path = WORKFLOW_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
    return {"status": "deleted", "name": name}


# ── Run History ──────────────────────────────────────────────────────────────

@app.get("/api/runs", dependencies=[Depends(verify_api_key)])
async def list_runs():
    """List all pipeline output directories with their stage logs."""
    output_dir = FOUNDRY_DIR / "output"
    if not output_dir.exists():
        return {"runs": []}

    # Detect which output dir the active pipeline is writing to
    active_dir = None
    if state.running and state.active_proc and state.active_proc.returncode is None:
        # Find the most recently modified _stage_*.log across all output dirs
        try:
            all_logs = sorted(output_dir.rglob("_stage_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            if all_logs:
                active_dir = all_logs[0].parent.name
        except Exception:
            pass

    runs = []
    for model_dir in sorted(output_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        if not model_dir.is_dir():
            continue
        logs = []
        for log_file in sorted(model_dir.glob("_stage_*.log"), reverse=True):
            logs.append({
                "name": log_file.name,
                "size": log_file.stat().st_size,
                "modified": log_file.stat().st_mtime,
                "live": state.running and model_dir.name == active_dir and log_file == sorted(model_dir.glob("_stage_*.log"), key=lambda p: p.stat().st_mtime)[-1],
            })
        # Also check for magicquant subdir logs
        mq_dir = model_dir / "magicquant"
        ggufs = []
        if mq_dir.exists():
            for gguf in mq_dir.glob("*.gguf"):
                ggufs.append({"name": gguf.name, "size_gb": round(gguf.stat().st_size / 1e9, 1)})

        # Check what artifacts exist
        has_lora = (model_dir / "lora_adapters" / "adapter_config.json").exists()
        has_merged = (model_dir / "merged_model").exists() and any((model_dir / "merged_model").glob("*.safetensors"))

        runs.append({
            "model": model_dir.name,
            "path": str(model_dir),
            "logs": logs,
            "ggufs": ggufs,
            "has_lora": has_lora,
            "has_merged": has_merged,
            "active": model_dir.name == active_dir,
        })
    return {"runs": runs, "pipeline_running": state.running}


@app.get("/api/runs/{model}/{logfile}", dependencies=[Depends(verify_api_key)])
async def get_run_log(model: str, logfile: str):
    """Read a specific log file from a run."""
    # Validate names to prevent path traversal
    if not re.match(r'^[\w\-.]+$', model) or not re.match(r'^_stage_\d+\.log$', logfile):
        raise HTTPException(status_code=400, detail="Invalid name")

    path = FOUNDRY_DIR / "output" / model / logfile
    if not path.exists():
        raise HTTPException(status_code=404, detail="Log not found")

    # Return last 2000 lines if file is huge, or full content if reasonable
    content = path.read_text(errors="replace")
    lines = content.split("\n")
    return {
        "model": model,
        "logfile": logfile,
        "total_lines": len(lines),
        "content": content if len(content) < 500_000 else "\n".join(lines[-2000:]),
        "truncated": len(content) >= 500_000,
    }


_pipeline_lock = asyncio.Lock()


@app.post("/api/run", dependencies=[Depends(verify_api_key)])
async def start_pipeline(cfg: RunRequest):
    """
    Launch the pipeline in a background task.

    Accepts a full RunRequest with per-stage config and an enabled_stages list.
    Returns an error if a pipeline is already in progress.
    Uses an asyncio.Lock to prevent race conditions from rapid concurrent requests.
    """
    if _pipeline_lock.locked():
        return {"error": "Pipeline is already running"}
    async with _pipeline_lock:
        if state.running:
            return {"error": "Pipeline is already running"}
        state.running = True
        for s in ALL_STAGES:
            state.stages[s] = StageStatus.PENDING
        state.progress = 0
        asyncio.create_task(run_pipeline(cfg))
    return {"status": "started"}


@app.post("/api/stop", dependencies=[Depends(verify_api_key)])
async def stop_pipeline():
    """Request a graceful pipeline stop. Kills active subprocess and sets running flag to False."""
    if not state.running:
        return {"error": "Pipeline is not running"}
    state.running = False
    if state.active_proc and state.active_proc.returncode is None:
        try:
            os.killpg(os.getpgid(state.active_proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    await state.log("Stop requested by user", "warn")
    await state.broadcast({"type": "pipeline_done"})
    return {"status": "stopping"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, token: str = Query(default="")):
    """
    WebSocket endpoint for real-time log streaming.

    Clients connect here to receive log lines, stage updates, and progress
    events as JSON messages. The connection stays open until the client
    disconnects.

    Authentication is via the ``token`` query parameter (e.g. ``/ws?token=...``).
    """
    if API_KEY and token != API_KEY:
        await ws.close(code=4001, reason="Invalid API key")
        return
    await ws.accept()
    state.ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, ConnectionResetError, RuntimeError, Exception):
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("FOUNDRY_UI_PORT", foundry_settings.ui_port))
    uvicorn.run(app, host="0.0.0.0", port=port)
