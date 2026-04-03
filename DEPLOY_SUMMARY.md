# Deploy Summary

Production hardening pass completed: 2026-04-03

## Repos Pushed

| Repo | Remote URL | Branch | Commit SHA | Tag |
|------|-----------|--------|------------|-----|
| MagicQuant | https://github.com/lucasmcoleman/MagicQuant.git | master | `4003226beae028271ead005563fd81bc9840d12a` | `prod-hardening/2026-04-03` |
| pipeline | https://github.com/lucasmcoleman/pipeline.git | master | `541b4a41db14fedf4745288fcc62dbb087ffb89d` | `prod-hardening/2026-04-03` |

Both repos pushed directly to `master` (no branch protection encountered).

## Disposition of External Directories

### unsloth_compiled_cache/
**Status: DELETED**

Contained compiled Unsloth trainer modules (UnslothSFTTrainer.py, etc.) with Python 3.12 bytecode cache. Confirmed unreferenced by either repository — only present in `pipeline/.gitignore`. The pipeline migrated away from Unsloth to custom fast loaders in commits `cc18026` and `2c017f8`.

### unsloth-env/
**Status: FULLY RESOLVED — DELETED**

The `pipeline/.venv` symlink skeleton has been replaced with a self-contained venv:
1. ROCm-specific packages (torch, triton, rocm, torchaudio, torchvision, torchao, bitsandbytes, xformers, cut-cross-entropy) were repacked as wheels from the unsloth-env installation and installed into the fresh venv
2. MagicQuant installed as editable package from `/server/programming/MagicQuant`
3. All remaining PyPI dependencies installed via `pip install -e ".[dev]"`
4. The `pipeline/MagicQuant/magicquant` symlink was removed (MagicQuant is now a proper pip package)
5. `unsloth-env/` has been deleted from the workspace
6. Repacked ROCm wheels preserved at `/tmp/rocm_wheels/` for future venv rebuilds

Verified: `torch 2.11.0a0+rocm7.11.0a20260106` loads with GPU (`Radeon 8060S Graphics`), all imports work, 18/18 tests pass, `pip check` reports no broken requirements.

## Key Changes Made

### Critical Bug Fixes
1. **MagicQuant tensor quantization bug**: Added dtype validation guard in `encode_to_ggml_bytes()` and the GGUF writer worker thread. Non-floating-point tensors are now rejected with a descriptive `ValueError` instead of producing silent corruption. Test coverage added (18 tests).
2. **pipeline hf_upload.py dry_run NameError**: Fixed reference to `files_to_upload` (should be `file_tuples`) that caused `NameError` whenever dry_run discovered files to upload.
3. **MagicQuant file_type metadata**: Fixed hardcoded `general.file_type = 1` that reported wrong quantization type to llama.cpp.

### Infrastructure Added
- **Configuration**: Pydantic-settings `BaseSettings` in both repos with environment variable support and `.env` file loading
- **Structured logging**: structlog integration with consistent schema and WebSocket callback support in pipeline
- **Retry logic**: tenacity decorators on HuggingFace API calls and llama.cpp subprocess calls
- **Service layer**: Extracted FastAPI business logic from route handlers into `core/services.py`
- **Packaging**: `pyproject.toml`, Dockerfile, docker-compose.yml, Makefile, README.md for both repos
- **Tests**: pytest suite for MagicQuant quantization guards (18 tests, all passing)

### Files Removed
- Hardcoded module-level constants from `fast_train_zeroclaw.py` and `fast_export.py`
- Hardcoded `VENV_PYTHON` path in `ui/app.py` (replaced with runtime detection)
- Hardcoded `DEVICE = torch.device("cuda:0")` (replaced with runtime detection)

## Verification Results

| Check | Status |
|-------|--------|
| MagicQuant syntax check (all .py files) | PASS |
| Pipeline syntax check (all .py files) | PASS |
| MagicQuant pytest (18 tests) | PASS |
| MagicQuant `pip install -e ".[dev]"` | PASS |
| Pipeline config module loads | PASS |
| Pipeline logging module loads | PASS |
| Pipeline services module loads | PASS |
| docker-compose.yml validation | PASS |
| No sensitive files in commits | PASS |
| No unsloth-env content in commits | PASS |

## Manual Action Items

### Resolved
1. ~~**Complete unsloth-env cleanup**~~ — DONE. Fresh venvs created in both repos with ROCm wheels repacked from unsloth-env. `unsloth-env/` deleted. MagicQuant symlink removed.
2. ~~**MagicQuant as proper dependency**~~ — DONE. Installed as editable package (`pip install -e ../MagicQuant`) in pipeline's venv. `pyproject.toml` declares `magicquant>=0.1.0` for portability.
3. ~~**UI authentication**~~ — DONE. Optional API key auth added via `PIPELINE_API_KEY` env var. Bearer token on REST, query param on WebSocket. Backward compatible when unset.

### Remaining
4. **Training data**: `data/zeroclaw_training_data.jsonl` is tracked in git. Consider moving to HuggingFace Datasets or S3 for production.
5. **ROCm PyTorch variant**: `pyproject.toml` declares `torch>=2.0.0` without specifying the ROCm variant. ROCm wheels are preserved at `/tmp/rocm_wheels/` for rebuilding venvs. For fresh installs on AMD, install these wheels first then `pip install -e ".[dev]"`.

## Security Issues Found

No hardcoded secrets were found in either repository. HF_TOKEN is properly sourced from environment variables or `~/.cache/huggingface/token`. No credential rotation is required.

## Documents Produced

| Document | Location | Purpose |
|----------|----------|---------|
| PIPELINE_MAP.md | pipeline/ | Full pipeline architecture and data flow |
| AUDIT_REPORT.md | MagicQuant/ | Code audit findings |
| AUDIT_REPORT.md | pipeline/ | Code audit findings |
| CHANGELOG.md | MagicQuant/ | Change documentation |
| CHANGELOG.md | pipeline/ | Change documentation |
| OPEN_QUESTIONS.md | pipeline/ | Unresolved items |
| DEPLOY_SUMMARY.md | pipeline/ | This document |
