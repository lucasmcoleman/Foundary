# Foundry Logic Audit

Audited 2026-04-05 against core/*.py and ui/app.py.


## Category A: Silent Wrong Results

### A1. pipeline.py validate_dataset() always returns True

- **File:line** -- `core/pipeline.py:334`
- **Severity** -- CRITICAL
- **Issue** -- `validate_dataset()` tracks failures in `all_ok` but the final `return` statement ignores it: it unconditionally returns `True`. When a local dataset has malformed JSON, missing fields, or is completely empty, the function logs errors but then reports success, and training proceeds with bad data.
- **Example** -- A JSONL file where every line is missing the `"messages"` key. `_validate_local_dataset` returns `False`, `all_ok` becomes `False`, error lines are logged, but `validate_dataset` returns `True`. `stage_training` passes the check and launches a GPU subprocess that either crashes or trains on an empty dataset.
- **Fix** -- Change line 334 from `return True` to `return all_ok`.

---

### A2. pipeline.py script generation: unescaped f-string interpolation of user-controlled config values

- **File:line** -- `core/pipeline.py:416` (and lines 432, 440, 481, 486, 488, 510, 575, 576, 577, 672, 673, 674, 683-691)
- **Severity** -- CRITICAL (code injection / wrong output)
- **Issue** -- The CLI `stage_training()` and `stage_export()` use triple-quoted f-strings to generate Python scripts, directly interpolating config values with bare `{tc.model_name}`, `{tc.lr_scheduler_type}`, `{tc.optim}`, `{base_model_id}`, `{artifacts.lora_dir}`, `{artifacts.merged_dir}`, etc. These are string-interpolated without `repr()`, so any value containing a quote character or backslash will break the generated script or inject arbitrary Python code. This is in contrast to `services.py`, which consistently uses `repr()` for string parameters.
- **Example** -- A model name containing a double quote (e.g., a local path like `/data/my"model/`) produces:
  ```python
  model, tokenizer = fast_load_quantized_model("/data/my"model/")
  ```
  which is a syntax error. A crafted value like `"; import os; os.system("rm -rf /"); x="` would execute arbitrary code.
- **Fix** -- Use `repr()` for all string values interpolated into the generated scripts, matching the pattern already used in `services.py`. For example: `fast_load_quantized_model({tc.model_name!r})`, `lr_scheduler_type={tc.lr_scheduler_type!r}`, etc.

---

### A3. fast_export.py writes index.json for single-shard models even though they do not have one

- **File:line** -- `core/fast_export.py:249-255`
- **Severity** -- MEDIUM
- **Issue** -- When the base model is a single-shard model (no `model.safetensors.index.json`), the code still writes a `model.safetensors.index.json` in the output directory. While this is technically valid (HF tooling accepts it), the `idx_metadata` will be empty `{}` and the output format diverges from the original, which could cause downstream confusion or break tools that check for exact format parity.
- **Example** -- Export a single-shard model. The merged output has `model.safetensors.index.json` pointing to the single shard, whereas the original had no index file. Some tools check for the index to decide multi-shard vs. single-shard loading.
- **Fix** -- Only write the index.json if the original model had one (i.e., when `os.path.exists(idx_path)` was true). For single-shard outputs, omit it.

---

### A4. pipeline.py did_training / did_heretic / did_magicquant flags are always True/non-None

- **File:line** -- `core/pipeline.py:1043-1045`
- **Severity** -- MEDIUM
- **Issue** -- `stage_upload()` sets `did_training=config.training is not None`, `did_heretic=config.heretic is not None`, `did_magicquant=config.magicquant is not None`. But `config.training` is always a `TrainingConfig` instance (it is a non-optional field with a default), so `did_training` is always `True` even when training was skipped. The model card will therefore always include training details sections even for export-only or quant-only runs. The UI path in `app.py:644` does this correctly by checking `"training" in enabled`.
- **Example** -- Run `python pipeline.py --model org/model --no-export --no-magicquant --upload-to user/repo` (training only). Then run with `--no-magicquant` but keep export. The upload model card still claims training, heretic, and magicquant sections based on config presence, not on what actually ran.
- **Fix** -- Track which stages actually ran (e.g., from the `results` dict) and pass those booleans to `HFUploadConfig`, or check the `enabled` set like the UI path does.

---

### A5. fast_train_zeroclaw.py skip_names matching is substring-based, catches false positives

- **File:line** -- `core/fast_train_zeroclaw.py:155-156`
- **Severity** -- MEDIUM
- **Issue** -- The quantization skip logic uses `any(s in full_name for s in skip_names)` and `any(s in full_name for s in ["embed", "norm"])`. The `"embed"` and `"norm"` checks are substring matches on the full dotted path. A model with a layer named `embedding_projection` or `prenorm_attention` would skip quantization of those layers unintentionally. More importantly, `"norm"` matches `"normal"`, `"normalize"`, etc.
- **Example** -- A model architecture with a `normalize_before` module. The substring `"norm"` matches it, so its Linear layers are kept in bf16 instead of being quantized to 4-bit. This wastes memory but does not corrupt results -- just suboptimal.
- **Fix** -- Use more precise matching: check the component name (`name`) rather than the full path, or use a regex like `(^|\.)(embed|norm)($|\.)` or check for `LayerNorm`/`RMSNorm` isinstance checks.

---

### A6. detect_response_template may strip valid template characters

- **File:line** -- `core/fast_train_zeroclaw.py:335`
- **Severity** -- MEDIUM
- **Issue** -- After finding the text between the last end-of-turn marker and the assistant content, the code does `.lstrip("\n")`. For models where the response template includes significant leading newlines (e.g., Llama 3's `\n\n` after `<|end_header_id|>`), this strips them. Since the template is used for completion-only loss masking, an incorrect template means the DataCollatorForCompletionOnlyLM cannot find the boundary, causing the entire sequence to be masked (zero loss / no learning) or the entire sequence to be unmasked (training on prompts too).
- **Example** -- A model whose chat template produces `<|end_header_id|>\n\nassistant` where the double newline is semantically significant. The `lstrip("\n")` removes both newlines, the template becomes `assistant`, which may match user turns too or may not be found at all.
- **Fix** -- Validate the detected template by tokenizing it and checking that it occurs exactly at the expected positions in a probe conversation. Log a warning if the template cannot be uniquely matched.


## Category B: Fragile Assumptions

### B1. pipeline.py uses Path.cwd() at script generation time, not execution time

- **File:line** -- `core/pipeline.py:412, 440, 520, 571, 585, 919`
- **Severity** -- HIGH
- **Issue** -- The generated subprocess scripts contain hardcoded `Path("{Path.cwd()}")` -- this evaluates `Path.cwd()` at the time `stage_training`/`stage_export`/`stage_heretic` runs (i.e., when the f-string is evaluated), not when the subprocess script runs. If the parent process's CWD is different from where the Foundry project lives (e.g., the script is invoked from a different directory), the `sys.path.insert(0, ...)` will point to the wrong location and imports will fail. The `services.py` version correctly uses `self.pipeline_root` instead.
- **Example** -- Run `cd /tmp && python /server/programming/Foundry/core/pipeline.py --model foo/bar`. The generated script will have `sys.path.insert(0, "/tmp/core")` which does not exist.
- **Fix** -- Use `Path(__file__).resolve().parent` (the location of pipeline.py) rather than `Path.cwd()` to compute the core path. Or better yet, delegate to `services.py` which already does this correctly.

---

### B2. Hardcoded CUDA device index 0

- **File:line** -- `core/fast_train_zeroclaw.py:49, 383-384, 415` and `core/fast_export.py:42`
- **Severity** -- MEDIUM
- **Issue** -- `DEVICE = get_device()` returns `cuda:0`, and the main() function directly calls `torch.cuda.get_device_name(0)` and `torch.cuda.get_device_properties(0)`. On a multi-GPU system (rare for the target APU but possible in testing), this always uses device 0. More critically, `get_device()` correctly checks `torch.cuda.is_available()` but the generated scripts in `pipeline.py:415` and `services.py:112` hardcode `DEVICE = torch.device('cuda:0')` unconditionally -- if CUDA is not available, the subprocess will crash immediately.
- **Example** -- Run on a CPU-only test machine. The fast loader gracefully falls back to CPU, but the pipeline.py generated training script crashes with `RuntimeError: Found no NVIDIA/AMD GPU`.
- **Fix** -- Use the same `get_device()` function in the generated scripts instead of hardcoding `cuda:0`.

---

### B3. services.py ExportService embeds has_lora as a Python literal, not a variable

- **File:line** -- `core/services.py:256`
- **Severity** -- MEDIUM
- **Issue** -- The generated script line is `lora_dir={repr(lora_source)} if {has_lora} else None`. The Python bool `has_lora` is interpolated as a string literal `True` or `False` into the script text. This works because Python's `True`/`False` are valid identifiers, but it is fragile -- if the parameter type ever changes to a non-bool truthy/falsy value, or if the variable name collides, the generated code breaks silently.
- **Example** -- If `has_lora` were passed as `1` instead of `True`, the generated code would be `lora_dir='...' if 1 else None`, which still works but is surprising. Not a current bug, but a maintenance hazard.
- **Fix** -- Use `repr(has_lora)` or resolve the conditional at generation time: `f"lora_dir={repr(lora_source if has_lora else None)},\n"`.

---

### B4. Composite model detection relies on single attribute check

- **File:line** -- `core/fast_train_zeroclaw.py:75`
- **Severity** -- MEDIUM
- **Issue** -- `_is_composite = hasattr(config, 'text_config')` assumes that any model with a `text_config` attribute is a composite/multimodal model whose weights need remapping. If a non-composite model happens to have a `text_config` attribute (e.g., a custom config class that stores text parameters there), the weight remapping logic would incorrectly strip the `model.language_model.` prefix and fail to match tensors.
- **Example** -- A custom model where the config class inherits from a base that adds `text_config` for another purpose. The loader attempts to remap keys, drops all tensors, and produces a model full of zeros.
- **Fix** -- Also check that the weight map actually contains `model.language_model.*` keys before enabling the composite path. The code at line 191 in `fast_export.py` does the right thing by checking `any(k.startswith(_composite_prefix) for k in weight_map)`.


## Category C: Race Conditions and State Bugs

### C1. PipelineState is a module-level singleton with no synchronization

- **File:line** -- `ui/app.py:106`
- **Severity** -- HIGH
- **Issue** -- `state = PipelineState()` is a module-level singleton. The `ws_clients` list is mutated from both `broadcast()` (inside the pipeline task) and the WebSocket endpoint (when clients connect/disconnect). While asyncio is single-threaded, the `broadcast()` method does `for ws in list(self.ws_clients)` and then modifies `ws_clients` by removing dead sockets. If a new WebSocket connects during a broadcast (between the `await ws.send_json()` calls), the newly appended client will not receive the current message (benign), but the `dead` list removal could remove the wrong entry if list indices shift. Additionally, `state.running` is set to `True` in both `start_pipeline` (line 862) and `run_pipeline` (line 752), creating a TOCTOU window.
- **Example** -- Two near-simultaneous `/api/run` POST requests. Both check `state.running` (False), both set it to True, both launch `asyncio.create_task(run_pipeline(cfg))`. Two pipeline tasks now run concurrently, writing to the same output directory and broadcasting interleaved logs to WebSocket clients.
- **Fix** -- Use an `asyncio.Lock` to serialize the running check and task creation. The `broadcast()` pattern is fine for asyncio but the double-set of `state.running` should be an atomic check-and-set.

---

### C2. start_pipeline resets all stage states before the background task starts

- **File:line** -- `ui/app.py:863-864`
- **Severity** -- MEDIUM
- **Issue** -- `start_pipeline` resets all stages to `PENDING` before `run_pipeline` starts executing in the background. `run_pipeline` then also resets stages at line 767. The first reset happens synchronously in the HTTP handler; the second happens asynchronously in the background task. If the user polls `/api/state` between these two events, they see `PENDING` for all stages even though no actual work has started yet (the background task has not yet reached its own stage-setting code).
- **Example** -- Client sends POST /api/run, gets response. Immediately polls GET /api/state. Sees all stages PENDING but `running=True`. The background task has not yet validated or started. The user sees a misleading "starting" state.
- **Fix** -- Only reset state inside `run_pipeline()`, or set stages to PENDING only after validation passes.

---

### C3. Stage skip logic uses artifact existence as "already done" -- no integrity check

- **File:line** -- `ui/app.py:377-382` (training skip), `428-442` (export skip), `506-511` (heretic skip), `554-562` (MagicQuant skip)
- **Severity** -- HIGH
- **Issue** -- Every stage checks for the existence of output artifacts and skips if found. For example, training checks for `lora_adapters/adapter_config.json`, export checks for any `*.safetensors` in `merged_model/`, MagicQuant checks for any `*.gguf`. There is no validation that these artifacts are complete or match the current config. A partial write (crash during training) could leave a truncated `adapter_config.json`, and the next run would skip training entirely. A previous run with different hyperparameters would also be silently reused.
- **Example** -- Training crashes mid-epoch. The `lora_adapters/` directory exists with `adapter_config.json` from PEFT's `save_pretrained()` call at the start, but the adapter weights are from a previous run or incomplete. The next pipeline invocation skips training, exports the stale/partial adapters, and produces a wrong model.
- **Fix** -- At minimum, check file sizes are non-zero and verify key files exist (not just `adapter_config.json` but also `adapter_model.safetensors`). Better: store a `_stage_complete.json` marker with a timestamp and config hash that is written only after successful completion, and check for that marker instead.

---

### C4. No cleanup of generated stage scripts on failure

- **File:line** -- `core/pipeline.py:516-520`, `582-585`, `916-919` and `ui/app.py:177-178`
- **Severity** -- LOW
- **Issue** -- Generated scripts (`_stage_train.py`, `_stage_export.py`, `_stage_heretic.py`, `_stage_<timestamp>.py`) are written to the output directory but never cleaned up. They accumulate across runs and contain interpolated config values (model paths, hyperparameters). In the UI path, each run creates a new timestamped script, so a long-running server accumulates many script files.
- **Example** -- After 50 pipeline runs, the output directory contains 200+ `_stage_*.py` files with embedded paths and config values.
- **Fix** -- Delete the script file after the subprocess completes (or at least on success). Alternatively, write to a tmpfile and clean up in a finally block.


## Category D: Security and Reliability

### D1. UI config endpoint accepts arbitrary JSON and persists it

- **File:line** -- `ui/app.py:843-849`
- **Severity** -- MEDIUM
- **Issue** -- `POST /api/config` accepts any JSON dict and merges it into the persisted `config.json` via `cfg.update(body)`. There is no validation of key names or value types. An attacker (or a buggy frontend) could inject arbitrary keys, overwrite existing keys with wrong types, or store very large values that bloat the config file.
- **Example** -- POST `{"hf_username": 12345, "__proto__": {"evil": true}, "giant_blob": "A" * 10000000}`. The config file grows to 10 MB and subsequent reads slow down.
- **Fix** -- Validate the incoming body against an expected schema (allowlist of keys, type checks) before merging.

---

### D2. Missing timeout on subprocess calls (CLI pipeline)

- **File:line** -- `core/pipeline.py:178-187` (`_run` function)
- **Severity** -- MEDIUM
- **Issue** -- `_run()` calls `proc.wait()` with no timeout. If a subprocess hangs (e.g., a GPU kernel deadlock, a download that stalls), the pipeline blocks forever with no way to recover short of killing the parent process. The UI path has a stop button that sends SIGTERM, but the CLI path has no such mechanism.
- **Example** -- `fast_train_zeroclaw.py` hits a GPU hang (HSA_ENABLE_SDMA=0 sometimes does not fully prevent these on gfx1151). The `_run()` call blocks indefinitely. The user must Ctrl+C the parent or kill -9.
- **Fix** -- Add a configurable timeout parameter to `_run()` and call `proc.wait(timeout=...)`. On timeout, kill the subprocess and return an error code.

---

### D3. Log file handle leak if subprocess raises during stdout reading

- **File:line** -- `ui/app.py:195-262`
- **Severity** -- LOW
- **Issue** -- The `run_script` function opens a log file at line 195 (`log_file = open(log_path, "w")`). The `finally` block at line 261-262 closes it. However, if the `asyncio.create_subprocess_exec` call itself raises (e.g., `FileNotFoundError` if the Python binary does not exist), the exception propagates before `proc` is assigned, and the outer `try/finally` closes the log file correctly. But if the exception occurs between the process creation and the `try` block entry at line 207, `state.active_proc` could be set to a failed process without being cleared. The code handles this correctly with the nested try/finally, so this is a minor note rather than a bug.
- **Fix** -- No immediate fix needed, but wrapping the entire subprocess lifecycle in a context manager would be cleaner.

---

### D4. UI and CLI pipeline.py have diverged in behavior

- **File:line** -- Multiple locations across `core/pipeline.py` and `ui/app.py`
- **Severity** -- HIGH
- **Issue** -- The CLI path (`pipeline.py:stage_training`, `stage_export`, `stage_heretic`) and the UI path (`app.py:do_training`, `do_export`, `do_heretic`) duplicate significant logic with subtle differences:
  - **Training**: CLI uses `prepare_model_for_kbit_training` (line 419); UI via `services.py` uses manual kbit prep (lines 116-125) with a different approach that avoids fp32 upcast of MoE experts. These produce different gradient behaviors.
  - **Dataset validation**: CLI's `validate_dataset` (pipeline.py:334) always returns True (bug A1 above); UI's `validate_dataset` (app.py:360) correctly returns `all_ok`.
  - **Output dir resolution**: CLI uses `Path.cwd()` for script paths; UI uses `FOUNDRY_DIR` / `FOUNDRY_ROOT`.
  - **Skip logic**: CLI has no skip-if-already-done logic; UI skips if artifacts exist.
  - **Config values**: CLI training script uses `warmup_ratio` (line 490); UI training script uses `warmup_steps` (services.py line 204). These are different parameters with different semantics in SFTConfig.
  - **Heretic**: CLI has a longer response prefix check list including `<|channel|>analysis<|message|>` (line 748); the services.py version starts at a different prefix list.
- **Example** -- Running the same config through the CLI vs. the UI produces different LoRA adapters because the kbit prep and warmup handling differ.
- **Fix** -- Consolidate all script generation into `services.py` and have both the CLI and UI call the same service classes. The current duplication is the root cause of most divergence bugs.

---

### D5. HF token not auto-loaded in CLI pipeline path

- **File:line** -- `core/pipeline.py:167-176` (the `_run` helper)
- **Severity** -- MEDIUM
- **Issue** -- The UI path's `run_script()` (app.py:189-192) auto-loads the HF token from `~/.cache/huggingface/token` if `HF_TOKEN` is not in the environment. The CLI path's `_run()` helper does not do this. If the user has logged in via `huggingface-cli login` but does not have `HF_TOKEN` in their shell environment, the CLI pipeline's upload stage and any HF Hub downloads requiring authentication will fail silently or with an unhelpful error.
- **Fix** -- Add the same token auto-loading logic to `_run()`:
  ```python
  if "HF_TOKEN" not in env:
      token_path = Path.home() / ".cache" / "huggingface" / "token"
      if token_path.exists():
          env["HF_TOKEN"] = token_path.read_text().strip()
  ```

---

### D6. hf_upload.py retry only catches requests.exceptions.RequestException

- **File:line** -- `core/hf_upload.py:401-419`
- **Severity** -- MEDIUM
- **Issue** -- The retry decorators on `_create_repo_with_retry`, `_upload_with_retry`, and `_whoami_with_retry` only retry on `requests.exceptions.RequestException`. The `huggingface_hub` library can also raise `huggingface_hub.utils.HfHubHTTPError` (which inherits from `requests.HTTPError`, so this is partially covered) and `huggingface_hub.utils.EntryNotFoundError` or `huggingface_hub.utils.RepositoryNotFoundError` (which do not inherit from `RequestException`). More importantly, transient server errors (HTTP 500, 502, 503) return `HfHubHTTPError`, and connection timeouts can raise `urllib3.exceptions.ReadTimeoutError` which is not a `RequestException` subclass in all configurations.
- **Example** -- HuggingFace Hub returns a 502 Bad Gateway during upload. The `huggingface_hub` client raises `HfHubHTTPError`. If the error chain does not include a `RequestException` base, the retry is not triggered and the upload fails immediately.
- **Fix** -- Add `huggingface_hub.utils.HfHubHTTPError` and `ConnectionError` to the retry exception types. Or use a broader `retry_if_exception(lambda e: ...)` predicate that catches HTTP 5xx errors.


## Summary

| Severity | Count | IDs |
|----------|-------|-----|
| CRITICAL | 2 | A1, A2 |
| HIGH | 4 | B1, C1, C3, D4 |
| MEDIUM | 10 | A3, A4, A5, A6, B2, B3, B4, C2, D2, D5, D6 |
| LOW | 2 | C4, D3 |

The two critical issues (A1 and A2) should be fixed immediately:
- A1 is a one-line fix (`return True` -> `return all_ok`) that restores dataset validation in the CLI path.
- A2 requires systematic use of `repr()` for all string interpolations in pipeline.py's generated scripts, matching the pattern already established in services.py.

The highest-leverage structural fix is D4 (consolidating script generation into services.py) because it eliminates the duplication that causes A1, A2, B1, and several other divergence issues simultaneously.
