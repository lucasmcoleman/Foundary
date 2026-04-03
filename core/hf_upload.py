"""
HuggingFace Hub upload module.

Handles repo creation, model card generation, and GGUF/LoRA/merged file upload
with progress reporting and dry-run mode.

Can be used standalone or called from pipeline.py / the web UI.

Token is sourced from HF_TOKEN env var — never hardcoded.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

LogFn = Callable[[str, str], None]


def _default_log(msg: str, level: str = "info"):
    prefix = {"error": "ERROR", "warn": "WARN", "success": "OK", "stage": ">>>"}.get(level, "   ")
    print(f"[{prefix}] {msg}")


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class HFUploadConfig:
    """All parameters needed for a HuggingFace upload."""
    repo_id: str = ""
    private: bool = True
    license: str = "apache-2.0"

    # What to upload
    upload_gguf: bool = True
    upload_lora: bool = False
    upload_merged: bool = False
    upload_dataset: bool = True

    # Model metadata (for the model card)
    base_model: str = ""
    dataset_name: str = ""
    model_description: str = ""

    # Pipeline stages that were actually run (for conditional card sections)
    did_training: bool = True
    did_heretic: bool = False
    did_magicquant: bool = True

    # Training details (for the model card)
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    num_epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_length: int = 8192
    batch_size: int = 2
    gradient_accumulation: int = 4
    optimizer: str = "adamw_8bit"
    lr_scheduler: str = "cosine"


# ── File discovery ───────────────────────────────────────────────────────────

def discover_upload_files(
    output_dir: str,
    upload_gguf: bool = True,
    upload_lora: bool = False,
    upload_merged: bool = False,
) -> list[tuple[Path, str]]:
    """Find files to upload from the output directory.

    Returns a list of (local_path, repo_path) tuples.
    """
    out = Path(output_dir)
    files = []

    if upload_lora:
        lora_dir = out / "lora_adapters"
        if lora_dir.exists():
            for f in sorted(lora_dir.iterdir()):
                if f.is_file():
                    files.append((f, f"lora/{f.name}"))

    if upload_merged:
        merged_dir = out / "merged_model"
        if merged_dir.exists():
            for f in sorted(merged_dir.iterdir()):
                if f.is_file():
                    files.append((f, f"merged/{f.name}"))

    if upload_gguf:
        gguf_files = []
        mq_dir = out / "magicquant"
        if mq_dir.exists():
            gguf_files = sorted(mq_dir.glob("*.gguf"))
        if not gguf_files:
            bf16 = out / "model-bf16.gguf"
            if bf16.exists():
                gguf_files = [bf16]
        for f in gguf_files:
            files.append((f, f.name))

    return files


# ── Model card generation ────────────────────────────────────────────────────

def generate_model_card(
    cfg: HFUploadConfig,
    files_to_upload: list[tuple[Path, str]],
    dataset_repo_id: str = "",
) -> str:
    """Generate a complete model card with YAML front matter.

    Includes: description, base model credit, quantization method,
    training details, caveats, limitations, and usage instructions.
    """
    repo_name = cfg.repo_id.split("/")[-1] if "/" in cfg.repo_id else cfg.repo_id
    base_model = cfg.base_model or "unknown"
    base_model_short = base_model.split("/")[-1] if "/" in base_model else base_model
    dataset_name = cfg.dataset_name or "custom dataset"

    # Build the GGUF file table
    gguf_rows = ""
    has_gguf = False
    for local_path, repo_path in files_to_upload:
        if repo_path.endswith(".gguf"):
            has_gguf = True
            size_gb = local_path.stat().st_size / 1e9
            name = repo_path
            # Infer quant tier from filename
            quant_hint = ""
            name_lower = name.lower()
            if "q4" in name_lower:
                quant_hint = "Q4 hybrid"
            elif "q5" in name_lower:
                quant_hint = "Q5 hybrid"
            elif "q6" in name_lower:
                quant_hint = "Q6 hybrid"
            elif "bf16" in name_lower:
                quant_hint = "BF16 (unquantized)"
            elif "f16" in name_lower:
                quant_hint = "F16 (unquantized)"
            gguf_rows += f"| [{name}](./{name}) | {size_gb:.1f} GB | {quant_hint} |\n"

    # Non-GGUF files table
    other_rows = ""
    for local_path, repo_path in files_to_upload:
        if not repo_path.endswith(".gguf"):
            size_mb = local_path.stat().st_size / 1e6
            other_rows += f"| {repo_path} | {size_mb:.0f} MB |\n"

    # Build a dynamic description based on which pipeline stages ran
    description = cfg.model_description
    if not description:
        parts = []
        if cfg.did_training:
            parts.append(f"fine-tuned on {dataset_name} with QLoRA")
        if cfg.did_heretic:
            parts.append("abliterated with [Heretic](https://github.com/p-e-w/heretic) for uncensored responses")
        if cfg.did_magicquant:
            parts.append("quantized using MagicQuant hybrid evolutionary per-tensor search")
        elif has_gguf:
            parts.append("exported to GGUF format")

        if parts:
            actions = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] if len(parts) > 1 else parts[0]
            description = (
                f"Derivative of [{base_model_short}](https://huggingface.co/{base_model}), "
                f"{actions}."
            )
        else:
            description = f"Derivative of [{base_model_short}](https://huggingface.co/{base_model})."

    # Build effective batch size
    effective_batch = cfg.batch_size * cfg.gradient_accumulation

    # Dynamic tags based on pipeline stages
    tags = []
    if has_gguf:
        tags.append("gguf")
    if cfg.did_magicquant:
        tags.extend(["quantized", "magicquant"])
    if cfg.did_training:
        tags.extend(["fine-tuned", "qlora"])
    if cfg.did_heretic:
        tags.extend(["abliterated", "uncensored", "heretic"])
    if not tags:
        tags = ["gguf"]
    tags_yaml = "\n".join(f"  - {t}" for t in tags)

    # Determine quantization types from GGUF filenames for metadata
    quant_types = set()
    for local_path, repo_path in files_to_upload:
        name = repo_path.lower()
        if name.endswith(".gguf"):
            for qt in ["q3", "q4", "q5", "q6", "q8", "mxfp4", "iq4", "bf16", "f16"]:
                if qt in name:
                    quant_types.add(qt.upper())

    is_finetune = cfg.did_training and cfg.num_epochs > 0 and cfg.lora_r > 0
    base_model_relation = "finetune" if is_finetune else "quantized"

    # Dynamic library_name and quantized_by
    library_name = "llama.cpp" if has_gguf else "transformers"
    quantized_by = "MagicQuant" if cfg.did_magicquant else ""

    yaml_block = f"""---
license: {cfg.license}
library_name: {library_name}
base_model:
  - {base_model}
base_model_relation: {base_model_relation}
{"datasets:" + chr(10) + "  - " + dataset_repo_id if dataset_repo_id else ""}
pipeline_tag: text-generation
{"quantized_by: " + quantized_by if quantized_by else ""}
language:
  - en
tags:
{tags_yaml}
---"""

    # Files section
    files_section = ""
    if has_gguf:
        files_section += f"""
## GGUF Files

| File | Size | Quant |
|------|------|-------|
{gguf_rows}"""

    if other_rows:
        files_section += f"""
## Other Files

| File | Size |
|------|------|
{other_rows}"""

    # ── Build the card body with conditional sections ──

    body_sections = []

    # Header
    body_sections.append(f"# {repo_name}\n\n{description}")

    # Base model credit (always)
    body_sections.append(f"""## Base Model

This is a derivative of [{base_model_short}](https://huggingface.co/{base_model}).
All credit for the base model architecture and weights goes to the original authors.
The base model's license applies to this derivative.""")

    # Heretic section (only if abliteration was applied)
    if cfg.did_heretic:
        body_sections.append("""## Abliteration (Heretic)

Safety alignment has been removed using **[Heretic](https://github.com/p-e-w/heretic)** directional ablation:

- A "refusal direction" is identified in each transformer layer's residual stream
- The model's projection matrices are orthogonalized with respect to that direction
- An Optuna TPE optimizer automatically tunes all ablation parameters
- The result is a model that responds to all prompts without refusal behavior

**This model will comply with any request.** Use responsibly.""")

    # MagicQuant section (only if quantized with MagicQuant)
    if cfg.did_magicquant:
        body_sections.append("""## Quantization Method

Quantized using **[MagicQuant](https://github.com/lucasmcoleman/MagicQuant)** hybrid evolutionary per-tensor quantization,
based on the methodology by **[magiccodingman](https://github.com/magiccodingman/MagicQuant-Wiki)**:

- Tensors are classified into sensitivity groups (Embeddings, Head, Query, Key, Output, FFN Up/Down, MoE Experts, Router)
- An evolutionary search finds the optimal quantization type per group, balancing size vs. perplexity
- **Q4/Q5/Q6 tier targets** are produced with different size-quality tradeoffs
- Small-row tensors and sensitivity-critical layers (embeddings, output head, router) are kept at F32/F16/BF16
- This is NOT a uniform quantization -- each tensor group gets its own optimal type""")

    # Training details (only if training was done)
    if cfg.did_training:
        dataset_link = f"[{Path(dataset_name).stem}](https://huggingface.co/datasets/{dataset_repo_id})" if dataset_repo_id else dataset_name
        body_sections.append(f"""## Training Details

| Parameter | Value |
|-----------|-------|
| Method | QLoRA with completion-only loss masking |
| LoRA rank (r) | {cfg.lora_r} |
| LoRA alpha | {cfg.lora_alpha} |
| LoRA dropout | {cfg.lora_dropout} |
| Epochs | {cfg.num_epochs} |
| Learning rate | {cfg.learning_rate} |
| LR scheduler | {cfg.lr_scheduler} |
| Batch size | {cfg.batch_size} (effective {effective_batch} with gradient accumulation) |
| Optimizer | {cfg.optimizer} |
| Training sequence length | {cfg.max_seq_length} |
| Precision | BF16 |
| Dataset | {dataset_link} |
| Hardware | AMD Ryzen AI Max+ 395 (Strix Halo), 128 GB unified memory (GTT), ROCm |

**Completion-only loss**: Only assistant response turns contribute to the training loss.
System and user turns are masked, so the model learns to generate responses rather
than memorizing prompts.""")

    # Files section
    if files_section:
        body_sections.append(files_section.strip())

    # Usage section (only if GGUF files present)
    if has_gguf:
        body_sections.append(f"""## Usage

### LM Studio

1. Download the GGUF file of your preferred quantization tier
2. Place it in your LM Studio models directory
3. Load the model in LM Studio -- it will auto-detect the chat template
4. The model supports the base model's full context length

### llama.cpp

```bash
# Interactive chat
llama-cli -m {repo_name}-Q5.gguf -c 8192 --chat-template chatml -cnv

# Single prompt
llama-cli -m {repo_name}-Q5.gguf -c 8192 -p "Your prompt here"

# Server mode
llama-server -m {repo_name}-Q5.gguf -c 8192 --port 8080
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(model_path="./{repo_name}-Q5.gguf", n_ctx=8192)
output = llm.create_chat_completion(
    messages=[
        {{"role": "user", "content": "Hello, how are you?"}}
    ]
)
print(output["choices"][0]["message"]["content"])
```""")

    # Caveats (dynamic)
    caveats = []
    caveats.append(f"- The base model's license ({cfg.license}) applies to all derivative files")
    if cfg.did_training:
        caveats.append("- This is a **personal fine-tune**, not an official release from the base model authors")
        caveats.append("- Quality depends on the training data and may not generalize to all tasks")
    if cfg.did_heretic:
        caveats.append("- **Safety alignment has been removed** -- this model may produce harmful, offensive, or dangerous content")
        caveats.append("- The abliteration process may slightly degrade model quality on some benchmarks")
    if cfg.did_magicquant or has_gguf:
        caveats.append("- Quantization reduces precision -- verify outputs for your specific use case")
    if cfg.did_magicquant:
        caveats.append("- The hybrid quantization assigns different precision to different tensor groups, which means quality characteristics may differ from uniform quantizations")
    body_sections.append("## Caveats\n\n" + "\n".join(caveats))

    # Limitations (dynamic)
    limitations = []
    if cfg.did_training:
        limitations.append(f"- Training data used sequences up to {cfg.max_seq_length} tokens; the model retains the base model's full context window")
        limitations.append("- Performance on tasks not represented in the training data may be degraded")
    if cfg.did_magicquant or has_gguf:
        limitations.append("- Quantized models may exhibit subtle differences from the full-precision fine-tune")
    limitations.append("- This model inherits any limitations and biases present in the base model")
    body_sections.append("## Limitations\n\n" + "\n".join(limitations))

    # Pipeline credit
    pipeline_parts = []
    if cfg.did_training:
        pipeline_parts.append("[Foundry](https://github.com/lucasmcoleman/Foundry)")
    if cfg.did_heretic:
        pipeline_parts.append("[Heretic](https://github.com/p-e-w/heretic)")
    if cfg.did_magicquant:
        pipeline_parts.append("[MagicQuant](https://github.com/lucasmcoleman/MagicQuant)")
    body_sections.append("---\n*Generated with " + " + ".join(pipeline_parts or ["Foundry"]) + "*")

    body = "\n\n".join(body_sections)
    card = f"{yaml_block}\n\n{body}\n"
    return card


# ── Retry wrappers ──────────────────────────────────────────────────────────


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=30),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def _create_repo_with_retry(api, **kwargs):
    """Create or verify a HuggingFace repo with automatic retry on transient failures."""
    return api.create_repo(**kwargs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=30),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def _upload_with_retry(api, **kwargs):
    """Upload a single file to HuggingFace with automatic retry on transient failures."""
    return api.upload_file(**kwargs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=30),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def _whoami_with_retry(api):
    """Validate HuggingFace token with automatic retry on transient failures."""
    return api.whoami()


# ── Dry-run mode ─────────────────────────────────────────────────────────────

@dataclass
class DryRunReport:
    """Result of a dry-run upload check."""
    token_valid: bool = False
    token_username: str = ""
    repo_accessible: bool = False
    repo_exists: bool = False
    repo_id: str = ""
    files: list[tuple[str, str, float]] = field(default_factory=list)  # (local, repo_path, size_gb)
    total_size_gb: float = 0.0
    model_card_preview: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.token_valid and self.repo_accessible and len(self.files) > 0 and not self.errors


def dry_run(
    cfg: HFUploadConfig,
    output_dir: str,
    log: LogFn = _default_log,
    token: Optional[str] = None,
) -> DryRunReport:
    """Validate credentials and repo access, report what would be uploaded.

    Does NOT upload anything.

    Args:
        cfg: Upload configuration
        output_dir: Directory containing artifacts to upload
        log: Logging callback
        token: HF token override (defaults to HF_TOKEN env var)

    Returns:
        DryRunReport with validation results and file list
    """
    report = DryRunReport(repo_id=cfg.repo_id)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        report.errors.append("huggingface_hub is not installed (pip install huggingface_hub)")
        log("huggingface_hub not installed", "error")
        return report

    # 1. Validate token
    log("Dry run: validating HF credentials", "stage")
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        report.errors.append("HF_TOKEN environment variable is not set")
        log("HF_TOKEN not set", "error")
        return report

    api = HfApi(token=hf_token)
    try:
        user_info = _whoami_with_retry(api)
        report.token_valid = True
        report.token_username = user_info.get("name", user_info.get("fullname", "unknown"))
        log(f"  Authenticated as: {report.token_username}")
    except Exception as e:
        report.errors.append(f"Token validation failed: {e}")
        log(f"Token validation failed: {e}", "error")
        return report

    # 2. Check repo access
    if not cfg.repo_id:
        report.errors.append("No repo_id configured")
        log("No repo_id configured", "error")
        return report

    log(f"  Checking repo: {cfg.repo_id}")
    try:
        api.repo_info(repo_id=cfg.repo_id, repo_type="model")
        report.repo_exists = True
        report.repo_accessible = True
        log(f"  Repo exists and is accessible")
    except Exception:
        # Repo doesn't exist yet -- check if we can create it
        report.repo_exists = False
        # Verify the namespace matches the authenticated user or their orgs
        namespace = cfg.repo_id.split("/")[0] if "/" in cfg.repo_id else report.token_username
        try:
            orgs = [o.get("name", "") for o in user_info.get("orgs", [])]
        except (AttributeError, TypeError):
            orgs = []

        if namespace == report.token_username or namespace in orgs:
            report.repo_accessible = True
            log(f"  Repo does not exist -- will be created on upload")
        else:
            report.repo_accessible = False
            report.errors.append(
                f"Cannot create repo under namespace '{namespace}' "
                f"(authenticated as '{report.token_username}', orgs: {orgs})"
            )
            log(f"  No access to namespace '{namespace}'", "error")

    # 3. Discover files
    log("  Scanning for uploadable files...")
    file_tuples = discover_upload_files(
        output_dir,
        upload_gguf=cfg.upload_gguf,
        upload_lora=cfg.upload_lora,
        upload_merged=cfg.upload_merged,
    )

    if not file_tuples:
        report.warnings.append("No files found to upload in the output directory")
        log("  No files found to upload", "warn")
    else:
        total = 0.0
        for local_path, repo_path in file_tuples:
            size_gb = local_path.stat().st_size / 1e9
            total += size_gb
            report.files.append((str(local_path), repo_path, size_gb))
            log(f"    {repo_path} ({size_gb:.2f} GB)")
        report.total_size_gb = total
        log(f"  Total upload size: {total:.2f} GB")

    # 4. Generate model card preview
    report.model_card_preview = generate_model_card(cfg, file_tuples)

    # Summary
    log("", "info")
    log("Dry run summary:", "stage")
    log(f"  Token valid:      {report.token_valid}")
    log(f"  User:             {report.token_username}")
    log(f"  Repo:             {report.repo_id}")
    log(f"  Repo exists:      {report.repo_exists}")
    log(f"  Repo accessible:  {report.repo_accessible}")
    log(f"  Files to upload:  {len(report.files)}")
    log(f"  Total size:       {report.total_size_gb:.2f} GB")
    if report.errors:
        for e in report.errors:
            log(f"  ERROR: {e}", "error")
    if report.warnings:
        for w in report.warnings:
            log(f"  WARNING: {w}", "warn")
    if report.ok:
        log("Dry run PASSED -- ready to upload", "success")
    else:
        log("Dry run FAILED -- see errors above", "error")

    return report


# ── Upload with progress ─────────────────────────────────────────────────────

def upload(
    cfg: HFUploadConfig,
    output_dir: str,
    log: LogFn = _default_log,
    token: Optional[str] = None,
) -> bool:
    """Upload artifacts to HuggingFace Hub with progress reporting.

    Args:
        cfg: Upload configuration
        output_dir: Directory containing artifacts to upload
        log: Logging callback (msg, level)
        token: HF token override (defaults to HF_TOKEN env var)

    Returns:
        True if all uploads succeeded
    """
    try:
        from huggingface_hub import HfApi, ModelCard
    except ImportError:
        log("huggingface_hub not installed (pip install huggingface_hub)", "error")
        return False

    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        log("HF_TOKEN environment variable is not set", "error")
        return False

    if not cfg.repo_id:
        log("No repo_id configured for upload", "error")
        return False

    api = HfApi(token=hf_token)

    # Validate credentials
    try:
        user_info = _whoami_with_retry(api)
        username = user_info.get("name", "unknown")
        log(f"Authenticated as: {username}")
    except Exception as e:
        log(f"Authentication failed: {e}", "error")
        return False

    # Create/verify repo
    log(f"Creating/verifying repo: {cfg.repo_id}", "stage")
    try:
        _create_repo_with_retry(
            api,
            repo_id=cfg.repo_id,
            repo_type="model",
            private=cfg.private,
            exist_ok=True,
        )
        log(f"  Repo ready: https://huggingface.co/{cfg.repo_id}")
    except Exception as e:
        log(f"Failed to create/access repo: {e}", "error")
        return False

    # Discover files
    files_to_upload = discover_upload_files(
        output_dir,
        upload_gguf=cfg.upload_gguf,
        upload_lora=cfg.upload_lora,
        upload_merged=cfg.upload_merged,
    )

    if not files_to_upload:
        log("No files found to upload", "warn")
        return False

    # Upload dataset as a separate HF dataset repo if configured (before model card so we can link it)
    dataset_repo_id = ""
    if cfg.upload_dataset and cfg.dataset_name:
        ds_path = Path(cfg.dataset_name)
        if not ds_path.is_absolute():
            ds_path = Path(output_dir).parent / cfg.dataset_name
            if not ds_path.exists():
                ds_path = Path(output_dir) / cfg.dataset_name
        if ds_path.exists() and ds_path.is_file():
            # Derive dataset repo name from model repo: "user/model-GGUF" -> "user/model-training-data"
            namespace = cfg.repo_id.split("/")[0] if "/" in cfg.repo_id else username
            model_short = cfg.repo_id.split("/")[-1] if "/" in cfg.repo_id else cfg.repo_id
            # Strip common suffixes to get a clean base name
            for suffix in ["-GGUF", "-MagicQuant-GGUF", "-gguf"]:
                if model_short.endswith(suffix):
                    model_short = model_short[:-len(suffix)]
                    break
            dataset_repo_id = f"{namespace}/{model_short}-training-data"

            log(f"Uploading dataset to {dataset_repo_id}", "stage")
            try:
                _create_repo_with_retry(
                    api,
                    repo_id=dataset_repo_id,
                    repo_type="dataset",
                    private=cfg.private,
                    exist_ok=True,
                )
                _upload_with_retry(
                    api,
                    path_or_fileobj=str(ds_path),
                    path_in_repo=ds_path.name,
                    repo_id=dataset_repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload training data ({ds_path.name})",
                )
                log(f"  Dataset uploaded to https://huggingface.co/datasets/{dataset_repo_id}", "success")
            except Exception as e:
                log(f"  Dataset upload failed (continuing): {e}", "warn")
                dataset_repo_id = ""

        # Remove dataset from the model file list — it goes to its own repo
        files_to_upload = [(p, r) for p, r in files_to_upload
                           if not r.startswith("training_data/")]

    total_size = sum(f.stat().st_size for f, _ in files_to_upload) / 1e9
    log(f"Found {len(files_to_upload)} files ({total_size:.2f} GB total)")

    # Generate and upload model card (after dataset upload so we have the repo ID to link)
    log("Generating model card", "stage")
    card_content = generate_model_card(cfg, files_to_upload, dataset_repo_id=dataset_repo_id)
    try:
        card = ModelCard(card_content)
        card.push_to_hub(cfg.repo_id, token=hf_token)
        log("  Model card uploaded", "success")
    except Exception as e:
        log(f"  Model card upload failed (continuing with files): {e}", "warn")

    # Upload files with progress
    log(f"Uploading {len(files_to_upload)} files", "stage")
    for i, (local_path, repo_path) in enumerate(files_to_upload, 1):
        size_gb = local_path.stat().st_size / 1e9
        log(f"  [{i}/{len(files_to_upload)}] {repo_path} ({size_gb:.2f} GB)")

        try:
            _upload_with_retry(
                api,
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=cfg.repo_id,
                repo_type="model",
                commit_message=f"Upload {repo_path}",
            )
            pct = int(100 * i / len(files_to_upload))
            log(f"    Uploaded ({pct}% complete)", "success")
        except Exception as e:
            log(f"    Failed to upload {repo_path}: {e}", "error")
            return False

    log(f"All files uploaded to https://huggingface.co/{cfg.repo_id}", "success")
    return True


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for standalone upload / dry-run."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload model artifacts to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (validate without uploading)
  python hf_upload.py --repo user/model-name --output-dir ./output --dry-run

  # Upload GGUF files
  python hf_upload.py --repo user/model-name --output-dir ./output

  # Upload everything (GGUF + LoRA + merged)
  python hf_upload.py --repo user/model-name --output-dir ./output --lora --merged

  # Show generated model card
  python hf_upload.py --repo user/model-name --output-dir ./output --dry-run --show-card
""",
    )
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID (user/model-name)")
    parser.add_argument("--output-dir", required=True, help="Pipeline output directory")
    parser.add_argument("--base-model", default="", help="Base model ID for model card")
    parser.add_argument("--dataset", default="", help="Dataset name for model card")
    parser.add_argument("--license", default="apache-2.0", help="License identifier")
    parser.add_argument("--private", action="store_true", help="Create as private repo")
    parser.add_argument("--public", action="store_true", help="Create as public repo")
    parser.add_argument("--lora", action="store_true", help="Upload LoRA adapters")
    parser.add_argument("--merged", action="store_true", help="Upload merged model")
    parser.add_argument("--no-gguf", action="store_true", help="Skip GGUF upload")
    parser.add_argument("--dry-run", action="store_true", help="Validate without uploading")
    parser.add_argument("--show-card", action="store_true", help="Print the generated model card")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (for model card)")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha (for model card)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (for model card)")
    parser.add_argument("--seq-length", type=int, default=8192, help="Max sequence length (for model card)")
    args = parser.parse_args()

    cfg = HFUploadConfig(
        repo_id=args.repo,
        private=not args.public if args.public else (args.private if args.private else True),
        license=args.license,
        upload_gguf=not args.no_gguf,
        upload_lora=args.lora,
        upload_merged=args.merged,
        base_model=args.base_model,
        dataset_name=args.dataset,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.epochs,
        max_seq_length=args.seq_length,
    )

    if args.dry_run:
        report = dry_run(cfg, args.output_dir)
        if args.show_card and report.model_card_preview:
            print("\n" + "=" * 60)
            print("MODEL CARD PREVIEW")
            print("=" * 60)
            print(report.model_card_preview)
        sys.exit(0 if report.ok else 1)
    else:
        ok = upload(cfg, args.output_dir)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
