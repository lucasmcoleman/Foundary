"""
Service layer for Foundry pipeline stages.

Extracts business logic from ui/app.py route handlers into testable,
reusable service classes. Each service accepts a config and a progress
callback, handles subprocess execution, and returns success/failure.

Usage from the UI:
    svc = TrainingService(project_root, venv_python)
    ok = await svc.run(cfg, run_script_fn)
"""

import json
import os
from pathlib import Path
from typing import Callable, Awaitable, Optional

# Type alias for the async subprocess runner used by the UI.
# Signature: (script_text, output_dir) -> exit_code
RunScriptFn = Callable[[str, str], Awaitable[int]]

# Type alias for the async log callback used by the UI.
AsyncLogFn = Callable[[str, str], Awaitable[None]]


def _env_preamble() -> str:
    """Return the standard ROCm environment setup block for subprocess scripts."""
    return (
        'import os\n'
        'os.environ["HSA_ENABLE_SDMA"] = "0"\n'
        'os.environ["PYTORCH_HIP_ALLOC_CONF"] = '
        '"backend:native,expandable_segments:True"\n'
        'os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"\n'
        'os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"\n'
    )


def _hf_cache_check(model_id_repr: str) -> str:
    """Return an HF cache probe snippet for subprocess scripts."""
    return (
        "from pathlib import Path as _P\n"
        f"_model_id = {model_id_repr}\n"
        "_is_local = _P(_model_id).exists()\n"
        "if _is_local:\n"
        '    print(f"Loading from local path: {_model_id}")\n'
        "else:\n"
        "    try:\n"
        "        from huggingface_hub import scan_cache_dir\n"
        "        _cached = False\n"
        "        for _repo in scan_cache_dir().repos:\n"
        "            if _repo.repo_id == _model_id:\n"
        "                _size_gb = _repo.size_on_disk / 1e9\n"
        '                print(f"Model found in HF cache '
        '({_size_gb:.1f} GB) — no download needed")\n'
        "                _cached = True\n"
        "                break\n"
        "        if not _cached:\n"
        '            print(f"Model not in cache — will download '
        'from HuggingFace: {_model_id}")\n'
        "    except Exception:\n"
        '        print(f"Loading model: {_model_id}")\n'
    )


class TrainingService:
    """Orchestrates the QLoRA training subprocess."""

    def __init__(self, pipeline_root: Path, venv_python: str) -> None:
        self.pipeline_root = pipeline_root
        self.venv_python = venv_python

    def build_script(
        self,
        *,
        model_name: str,
        dataset_path: str,
        output_dir: str,
        max_seq_length: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        use_rslora: bool,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        lr_scheduler_type: str,
        warmup_steps: int,
        optim: str,
    ) -> str:
        """Generate the training subprocess script text."""
        core_path = repr(str(self.pipeline_root / "core"))
        pipeline_root_str = repr(str(self.pipeline_root))

        script = _env_preamble()
        script += _hf_cache_check(repr(model_name))
        script += (
            f"\nimport sys, torch\n"
            f"sys.path.insert(0, str(_P({core_path})))\n"
            f"\n"
            f"from fast_train_zeroclaw import "
            f"fast_load_quantized_model, find_latest_checkpoint\n"
            f"from datasets import load_dataset\n"
            f"from trl import SFTTrainer, SFTConfig\n"
            f"from peft import LoraConfig, get_peft_model\n"
            f"\n"
            f"DEVICE = torch.device('cuda:0')\n"
            f"model, tokenizer = fast_load_quantized_model({repr(model_name)})\n"
            f"\n"
            f"# Manual kbit prep — avoids fp32 upcast of MoE experts\n"
            f"for param in model.parameters():\n"
            f"    param.requires_grad = False\n"
            f"for name, module in model.named_modules():\n"
            f'    if "norm" in name.lower() or "layernorm" in name.lower():\n'
            f"        module.to(torch.float32)\n"
            f"model.gradient_checkpointing_enable("
            f'gradient_checkpointing_kwargs={{"use_reentrant": False}})\n'
            f"model.enable_input_require_grads()\n"
            f'print(f"After manual kbit prep: '
            f'{{torch.cuda.memory_allocated()/1e9:.1f}} GB")\n'
            f"\n"
            f"lora_config = LoraConfig(\n"
            f"    r={lora_r}, lora_alpha={lora_alpha}, "
            f"lora_dropout={lora_dropout},\n"
            f'    target_modules=["q_proj","k_proj","v_proj","o_proj",'
            f'"gate_proj","up_proj","down_proj"],\n'
            f"    use_rslora={use_rslora}, task_type=\"CAUSAL_LM\",\n"
            f")\n"
            f"model = get_peft_model(model, lora_config)\n"
            f"\n"
            f"trainable = sum(p.numel() for p in model.parameters() "
            f"if p.requires_grad)\n"
            f"total = sum(p.numel() for p in model.parameters())\n"
            f'print(f"Trainable: {{trainable:,}} / {{total:,}} '
            f'({{100*trainable/total:.2f}}%)")\n'
            f"\n"
            f"_dataset_path = _P({repr(dataset_path)})\n"
            f"if not _dataset_path.is_absolute():\n"
            f"    _dataset_path = _P({pipeline_root_str}) / _dataset_path\n"
            f'dataset = load_dataset("json", '
            f'data_files=str(_dataset_path), split="train")\n'
            f'print(f"Dataset: {{len(dataset)}} examples")\n'
            f"\n"
            f"def fmt(ex):\n"
            f'    ex["text"] = tokenizer.apply_chat_template('
            f'ex["messages"], tokenize=False, '
            f"add_generation_prompt=False)\n"
            f"    return ex\n"
            f"dataset = dataset.map(fmt)\n"
            f"\n"
            f"resume_ckpt = find_latest_checkpoint({repr(output_dir)})\n"
            f"if resume_ckpt:\n"
            f'    print(f"Resuming from checkpoint: {{resume_ckpt}}")\n'
            f"\n"
            f"trainer = SFTTrainer(\n"
            f"    model=model, processing_class=tokenizer, "
            f"train_dataset=dataset,\n"
            f"    args=SFTConfig(\n"
            f"        output_dir={repr(output_dir)},\n"
            f"        num_train_epochs={num_train_epochs},\n"
            f"        per_device_train_batch_size="
            f"{per_device_train_batch_size},\n"
            f"        gradient_accumulation_steps="
            f"{gradient_accumulation_steps},\n"
            f"        learning_rate={learning_rate},\n"
            f"        lr_scheduler_type={repr(lr_scheduler_type)},\n"
            f"        warmup_steps={warmup_steps}, "
            f"optim={repr(optim)},\n"
            f"        weight_decay=0.01, max_grad_norm=1.0, "
            f"fp16=False, bf16=True,\n"
            f"        logging_steps=1, save_strategy=\"epoch\", "
            f"seed=42,\n"
            f"        gradient_checkpointing=True,\n"
            f"        gradient_checkpointing_kwargs="
            f'{{\"use_reentrant\": False}},\n'
            f'        report_to="none",\n'
            f"        max_length={max_seq_length},\n"
            f'        dataset_text_field="text",\n'
            f"        completion_only_loss=True,\n"
            f"    ),\n"
            f")\n"
            f"stats = trainer.train(resume_from_checkpoint=resume_ckpt)\n"
            f'print(f"Final loss: {{stats.training_loss:.4f}}")\n'
            f"\n"
            f"lora_dir = {repr(output_dir + '/lora_adapters')}\n"
            f"model.save_pretrained(lora_dir)\n"
            f"tokenizer.save_pretrained(lora_dir)\n"
            f'print("PIPELINE_STAGE_COMPLETE=training")\n'
        )
        return script


class ExportService:
    """Orchestrates the streaming LoRA merge / export subprocess."""

    def __init__(self, pipeline_root: Path, venv_python: str) -> None:
        self.pipeline_root = pipeline_root
        self.venv_python = venv_python

    def build_script(
        self,
        *,
        base_model_id: str,
        lora_source: Optional[str],
        has_lora: bool,
        merged_dir: str,
    ) -> str:
        """Generate the export subprocess script text."""
        core_path = repr(str(self.pipeline_root / "core"))

        script = _env_preamble()
        script += _hf_cache_check(repr(base_model_id))
        script += (
            f"\nimport sys\n"
            f"sys.path.insert(0, str(_P({core_path})))\n"
            f"from fast_export import streaming_merge\n"
            f"streaming_merge(\n"
            f"    model_id={repr(base_model_id)},\n"
            f"    lora_dir={repr(lora_source)} if {has_lora} else None,\n"
            f"    merged_dir={repr(merged_dir)},\n"
            f")\n"
            f'print("PIPELINE_STAGE_COMPLETE=export")\n'
        )
        return script


class HereticService:
    """Orchestrates the heretic abliteration subprocess."""

    def __init__(self, pipeline_root: Path, venv_python: str) -> None:
        self.pipeline_root = pipeline_root
        self.venv_python = venv_python

    def build_script(
        self,
        *,
        model_path: str,
        output_path: str,
        checkpoint_dir: str,
        n_trials: int,
        n_startup_trials: int,
        quantization: str,
        kl_divergence_scale: float,
        orthogonalize_direction: bool,
        row_normalization: str,
    ) -> str:
        """Generate the heretic abliteration subprocess script text.

        Uses heretic's internal API directly (Model, Evaluator, Optuna study)
        to bypass the interactive CLI prompts (questionary).
        """
        script = _env_preamble()
        script += (
            f"\nimport math, os, sys, time, warnings\n"
            f"import torch\n"
            f"import torch.nn.functional as F\n"
            f"import optuna\n"
            f"import transformers\n"
            f"from optuna.samplers import TPESampler\n"
            f"from optuna.storages import JournalStorage\n"
            f"from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock\n"
            f"from optuna.study import StudyDirection\n"
            f"from optuna.trial import TrialState\n"
            f"from optuna.exceptions import ExperimentalWarning\n"
            f"from optuna import Trial, TrialPruned\n"
            f"from dataclasses import asdict\n"
            f"from os.path import commonprefix\n"
            f"\n"
            f"from heretic.config import Settings\n"
            f"from heretic.model import Model, AbliterationParameters\n"
            f"from heretic.evaluator import Evaluator\n"
            f"from heretic.utils import (\n"
            f"    empty_cache, format_duration, get_trial_parameters,\n"
            f"    load_prompts, print_memory_usage,\n"
            f")\n"
            f"\n"
            f"# Use plain print for pipeline logging\n"
            f"import builtins\n"
            f"_print = builtins.print\n"
            f"from heretic import utils as _hu\n"
            f"_hu.print = _print\n"
            f"\n"
            f"torch.set_grad_enabled(False)\n"
            f"torch._dynamo.config.cache_size_limit = 64\n"
            f"transformers.logging.set_verbosity_error()\n"
            f"optuna.logging.set_verbosity(optuna.logging.WARNING)\n"
            f"warnings.filterwarnings('ignore', category=ExperimentalWarning)\n"
            f"\n"
            f"model_path = {repr(model_path)}\n"
            f"output_path = {repr(output_path)}\n"
            f"checkpoint_dir = {repr(checkpoint_dir)}\n"
            f"os.makedirs(checkpoint_dir, exist_ok=True)\n"
            f"\n"
            f"_print(f'Model: {{model_path}}')\n"
            f"_print(f'Output: {{output_path}}')\n"
            f"\n"
            f"settings = Settings(\n"
            f"    model=model_path,\n"
            f"    quantization={repr(quantization)},\n"
            f"    n_trials={n_trials},\n"
            f"    n_startup_trials={n_startup_trials},\n"
            f"    kl_divergence_scale={kl_divergence_scale},\n"
            f"    orthogonalize_direction={orthogonalize_direction},\n"
            f"    row_normalization={repr(row_normalization)},\n"
            f"    study_checkpoint_dir=checkpoint_dir,\n"
            f"    batch_size=0,\n"
            f")\n"
            f"\n"
            f"model = Model(settings)\n"
            f"_print()\n"
            f"print_memory_usage()\n"
            f"\n"
            f"_print()\n"
            f"_print(f'Loading good prompts from {{settings.good_prompts.dataset}}...')\n"
            f"good_prompts = load_prompts(settings, settings.good_prompts)\n"
            f"_print(f'  {{len(good_prompts)}} prompts loaded')\n"
            f"\n"
            f"_print()\n"
            f"_print(f'Loading bad prompts from {{settings.bad_prompts.dataset}}...')\n"
            f"bad_prompts = load_prompts(settings, settings.bad_prompts)\n"
            f"_print(f'  {{len(bad_prompts)}} prompts loaded')\n"
            f"\n"
            f"# Auto batch size\n"
            f"if settings.batch_size == 0:\n"
            f"    _print()\n"
            f"    _print('Determining optimal batch size...')\n"
            f"    batch_size = 1\n"
            f"    best_batch_size = -1\n"
            f"    best_performance = -1\n"
            f"    while batch_size <= settings.max_batch_size:\n"
            f"        _print(f'  Trying batch size {{batch_size}}... ', end='')\n"
            f"        prompts = good_prompts * math.ceil(batch_size / len(good_prompts))\n"
            f"        prompts = prompts[:batch_size]\n"
            f"        try:\n"
            f"            model.get_responses(prompts)\n"
            f"            st = time.perf_counter()\n"
            f"            responses = model.get_responses(prompts)\n"
            f"            et = time.perf_counter()\n"
            f"        except Exception as error:\n"
            f"            if batch_size == 1: raise\n"
            f"            _print(f'Failed ({{error}})')\n"
            f"            break\n"
            f"        rl = [len(model.tokenizer.encode(r)) for r in responses]\n"
            f"        perf = sum(rl) / (et - st)\n"
            f"        _print(f'Ok ({{perf:.0f}} tokens/s)')\n"
            f"        if perf > best_performance:\n"
            f"            best_batch_size = batch_size\n"
            f"            best_performance = perf\n"
            f"        batch_size *= 2\n"
            f"    settings.batch_size = best_batch_size\n"
            f"    _print(f'  Chosen batch size: {{settings.batch_size}}')\n"
            f"\n"
            f"# Check for common response prefix\n"
            f"_print()\n"
            f"_print('Checking for common response prefix...')\n"
            f"responses = model.get_responses_batched(good_prompts[:100] + bad_prompts[:100])\n"
            f"model.response_prefix = commonprefix(responses).rstrip(' ')\n"
            f"if model.response_prefix.startswith('<think>'):\n"
            f"    model.response_prefix = '<think></think>'\n"
            f"elif model.response_prefix.startswith('<thought>'):\n"
            f"    model.response_prefix = '<thought></thought>'\n"
            f"elif model.response_prefix.startswith('[THINK]'):\n"
            f"    model.response_prefix = '[THINK][/THINK]'\n"
            f"if model.response_prefix:\n"
            f"    _print(f'  Prefix found: {{model.response_prefix!r}}')\n"
            f"else:\n"
            f"    _print('  None found')\n"
            f"\n"
            f"evaluator = Evaluator(settings, model)\n"
            f"\n"
            f"# Compute refusal directions\n"
            f"_print()\n"
            f"_print('Calculating per-layer refusal directions...')\n"
            f"_print('  Obtaining residuals for good prompts...')\n"
            f"good_residuals = model.get_residuals_batched(good_prompts)\n"
            f"_print('  Obtaining residuals for bad prompts...')\n"
            f"bad_residuals = model.get_residuals_batched(bad_prompts)\n"
            f"good_means = good_residuals.mean(dim=0)\n"
            f"bad_means = bad_residuals.mean(dim=0)\n"
            f"refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)\n"
            f"if settings.orthogonalize_direction:\n"
            f"    good_directions = F.normalize(good_means, p=2, dim=1)\n"
            f"    proj = torch.sum(refusal_directions * good_directions, dim=1)\n"
            f"    refusal_directions = refusal_directions - proj.unsqueeze(1) * good_directions\n"
            f"    refusal_directions = F.normalize(refusal_directions, p=2, dim=1)\n"
            f"del good_residuals, bad_residuals\n"
            f"empty_cache()\n"
            f"\n"
            f"# Set up Optuna study\n"
            f"study_file = os.path.join(\n"
            f"    checkpoint_dir,\n"
            f"    ''.join([(c if (c.isalnum() or c in ['_', '-']) else '--') for c in settings.model]) + '.jsonl',\n"
            f")\n"
            f"lock_obj = JournalFileOpenLock(study_file)\n"
            f"backend = JournalFileBackend(study_file, lock_obj=lock_obj)\n"
            f"storage = JournalStorage(backend)\n"
            f"\n"
            f"trial_index = 0\n"
            f"start_index = 0\n"
            f"start_time = time.perf_counter()\n"
            f"\n"
            f"def objective(trial):\n"
            f"    global trial_index\n"
            f"    trial_index += 1\n"
            f"    trial.set_user_attr('index', trial_index)\n"
            f"    direction_scope = trial.suggest_categorical('direction_scope', ['global', 'per layer'])\n"
            f"    last_layer = len(model.get_layers()) - 1\n"
            f"    direction_index = trial.suggest_float('direction_index', 0.4 * last_layer, 0.9 * last_layer)\n"
            f"    if direction_scope == 'per layer': direction_index = None\n"
            f"    parameters = {{}}\n"
            f"    for comp in model.get_abliterable_components():\n"
            f"        mw = trial.suggest_float(f'{{comp}}.max_weight', 0.8, 1.5)\n"
            f"        mwp = trial.suggest_float(f'{{comp}}.max_weight_position', 0.6 * last_layer, 1.0 * last_layer)\n"
            f"        mnw = trial.suggest_float(f'{{comp}}.min_weight', 0.0, 1.0)\n"
            f"        mwd = trial.suggest_float(f'{{comp}}.min_weight_distance', 1.0, 0.6 * last_layer)\n"
            f"        parameters[comp] = AbliterationParameters(\n"
            f"            max_weight=mw, max_weight_position=mwp,\n"
            f"            min_weight=(mnw * mw), min_weight_distance=mwd,\n"
            f"        )\n"
            f"    trial.set_user_attr('direction_index', direction_index)\n"
            f"    trial.set_user_attr('parameters', {{k: asdict(v) for k, v in parameters.items()}})\n"
            f"    _print(f'\\nRunning trial {{trial_index}} of {{settings.n_trials}}...')\n"
            f"    model.reset_model()\n"
            f"    model.abliterate(refusal_directions, direction_index, parameters)\n"
            f"    score, kl_div, refusals = evaluator.get_score()\n"
            f"    elapsed = time.perf_counter() - start_time\n"
            f"    remaining = (elapsed / (trial_index - start_index)) * (settings.n_trials - trial_index)\n"
            f"    _print(f'  Elapsed: {{format_duration(elapsed)}}')\n"
            f"    if trial_index < settings.n_trials:\n"
            f"        _print(f'  Estimated remaining: {{format_duration(remaining)}}')\n"
            f"    trial.set_user_attr('kl_divergence', kl_div)\n"
            f"    trial.set_user_attr('refusals', refusals)\n"
            f"    return score\n"
            f"\n"
            f"def objective_wrapper(trial):\n"
            f"    try: return objective(trial)\n"
            f"    except KeyboardInterrupt:\n"
            f"        trial.study.stop()\n"
            f"        raise TrialPruned()\n"
            f"\n"
            f"study = optuna.create_study(\n"
            f"    sampler=TPESampler(n_startup_trials=settings.n_startup_trials, n_ei_candidates=128, multivariate=True),\n"
            f"    directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],\n"
            f"    storage=storage, study_name='heretic', load_if_exists=True,\n"
            f")\n"
            f"study.set_user_attr('settings', settings.model_dump_json())\n"
            f"study.set_user_attr('finished', False)\n"
            f"\n"
            f"def count_done(): return sum(1 for t in study.trials if t.state == TrialState.COMPLETE)\n"
            f"\n"
            f"start_index = trial_index = count_done()\n"
            f"if start_index > 0:\n"
            f"    _print(f'\\nResuming existing study ({{start_index}} trials completed).')\n"
            f"remaining = settings.n_trials - count_done()\n"
            f"if remaining > 0:\n"
            f"    study.optimize(objective_wrapper, n_trials=remaining)\n"
            f"if count_done() == settings.n_trials:\n"
            f"    study.set_user_attr('finished', True)\n"
            f"\n"
            f"# Select best trial from Pareto front\n"
            f"completed = [t for t in study.trials if t.state == TrialState.COMPLETE]\n"
            f"if not completed:\n"
            f"    _print('No completed trials')\n"
            f"    sys.exit(1)\n"
            f"sorted_trials = sorted(completed, key=lambda t: (t.user_attrs['refusals'], t.user_attrs['kl_divergence']))\n"
            f"min_div = math.inf\n"
            f"best_trials = []\n"
            f"for t in sorted_trials:\n"
            f"    kl = t.user_attrs['kl_divergence']\n"
            f"    if kl < min_div:\n"
            f"        min_div = kl\n"
            f"        best_trials.append(t)\n"
            f"best = best_trials[0]\n"
            f"_print(f'\\nSelected trial {{best.user_attrs[\"index\"]}}: '\n"
            f"       f'refusals={{best.user_attrs[\"refusals\"]}}, '\n"
            f"       f'KL={{best.user_attrs[\"kl_divergence\"]:.4f}}')\n"
            f"\n"
            f"# Apply best trial and save\n"
            f"_print('Resetting model...')\n"
            f"model.reset_model()\n"
            f"_print('Abliterating with best parameters...')\n"
            f"model.abliterate(\n"
            f"    refusal_directions,\n"
            f"    best.user_attrs['direction_index'],\n"
            f"    {{k: AbliterationParameters(**v) for k, v in best.user_attrs['parameters'].items()}},\n"
            f")\n"
            f"_print('Saving merged abliterated model...')\n"
            f"merged_model = model.get_merged_model()\n"
            f"merged_model.save_pretrained(output_path)\n"
            f"del merged_model\n"
            f"empty_cache()\n"
            f"model.tokenizer.save_pretrained(output_path)\n"
            f"_print(f'Abliterated model saved to {{output_path}}')\n"
            f"_print('PIPELINE_STAGE_COMPLETE=heretic')\n"
        )
        return script


class MagicQuantService:
    """Orchestrates the MagicQuant evolutionary search subprocess."""

    def __init__(self, pipeline_root: Path, venv_python: str) -> None:
        self.pipeline_root = pipeline_root
        self.venv_python = venv_python

    def build_script(
        self,
        *,
        llamacpp_hint: str,
        pipeline_root_str: str,
        mq_source_override: str,
        out_abs_str: str,
        generations: int,
        population_size: int,
        target_base_quant: str,
        tiers_json: str,
        model_name: str,
    ) -> str:
        """Generate the MagicQuant subprocess script text."""
        script = (
            "import sys, os, subprocess, multiprocessing\n"
            "from pathlib import Path\n"
            "\n"
            "def find_llamacpp():\n"
            f"    for p in [{repr(llamacpp_hint)}, "
            f'os.environ.get("LLAMACPP_PATH",""),\n'
            f'              str(Path.home()/"llama.cpp"), '
            f'"./llama.cpp", "/usr/local"]:\n'
            f"        if not p: continue\n"
            f"        pp = Path(p)\n"
            f'        for sub in [pp/"convert_hf_to_gguf.py", '
            f'pp/"build"/"bin"/"llama-quantize"]:\n'
            f"            if sub.exists(): return str(pp)\n"
            f"    return None\n"
            f"\n"
            f"llamacpp = find_llamacpp()\n"
            f"if not llamacpp:\n"
            f'    install_dir = Path.home() / "llama.cpp"\n'
            f'    print("llama.cpp not found — auto-installing...")\n'
            f'    rc = subprocess.run(["git", "clone", "--depth", "1",\n'
            f'                         '
            f'"https://github.com/ggml-org/llama.cpp.git", '
            f"str(install_dir)]).returncode\n"
            f"    if rc == 0:\n"
            f'        build_dir = install_dir / "build"\n'
            f'        rc = subprocess.run(["cmake", "-B", '
            f'str(build_dir), "-DCMAKE_BUILD_TYPE=Release",\n'
            f"                             "
            f"str(install_dir)]).returncode\n"
            f"        if rc == 0:\n"
            f"            jobs = str(multiprocessing.cpu_count())\n"
            f'            rc = subprocess.run(["cmake", "--build", '
            f'str(build_dir), "-j", jobs]).returncode\n'
            f"    if rc == 0:\n"
            f"        llamacpp = str(install_dir)\n"
            f'        print(f"llama.cpp installed: {{llamacpp}}")\n'
            f"    else:\n"
            f'        print("Warning: llama.cpp install failed, '
            f'using heuristic probing")\n'
            f"        llamacpp = None\n"
            f"\n"
            f"print(f\"llama.cpp: "
            f"{{llamacpp or 'not found (heuristic mode)'}}\")\n"
            f"\n"
            f"sys.path.insert(0, str(Path({repr(pipeline_root_str)}) "
            f'/ "MagicQuant"))\n'
            f"from magicquant.orchestrator import "
            f"MagicQuantOrchestrator\n"
            f"import json\n"
            f"\n"
            f"override = {repr(mq_source_override)}\n"
            f"out_dir = Path({repr(out_abs_str)})\n"
            f"\n"
            f"def _resolve_source(override, out_dir):\n"
            f"    candidates = []\n"
            f"    if override:\n"
            f"        p = Path(override)\n"
            f"        if not p.is_absolute():\n"
            f"            candidates = [out_dir / override, "
            f"Path({repr(pipeline_root_str)}) / override]\n"
            f"        else:\n"
            f"            candidates = [p]\n"
            f"    if not candidates:\n"
            f"        candidates = [out_dir]\n"
            f"    for c in candidates:\n"
            f"        if c.is_dir():\n"
            f'            heretic = c / "heretic_model"\n'
            f"            if heretic.exists() and "
            f'any(heretic.glob("*.safetensors")):\n'
            f"                return str(heretic)\n"
            f'            merged = c / "merged_model"\n'
            f"            if merged.exists() and "
            f'any(merged.glob("*.safetensors")):\n'
            f"                return str(merged)\n"
            f'            if any(c.glob("*.safetensors")):\n'
            f"                return str(c)\n"
            f'            gguf = c / "model-bf16.gguf"\n'
            f"            if gguf.exists():\n"
            f"                return str(gguf)\n"
            f"        elif c.is_file():\n"
            f"            return str(c)\n"
            f"    return None\n"
            f"\n"
            f"source = _resolve_source(override, out_dir)\n"
            f"if not source:\n"
            f'    print("Error: no source model found. Enable Export '
            f'or set a Source Model path in MagicQuant config.")\n'
            f"    sys.exit(1)\n"
            f'print(f"MagicQuant source: {{source}}")\n'
            f"\n"
            f"orch = MagicQuantOrchestrator(\n"
            f"    source_model_path=source,\n"
            f'    output_dir=str(out_dir / "magicquant"),\n'
            f"    llamacpp_path=llamacpp,\n"
            f")\n"
            f"\n"
            f'print(f"Search: generations={generations}, '
            f"population={population_size}, "
            f'base={target_base_quant}")\n'
            f"\n"
            f"best_configs, tiered = orch.run_full_search(\n"
            f"    target_base_quant={repr(target_base_quant)},\n"
            f"    max_generations={generations},\n"
            f"    population_size={population_size},\n"
            f"    verbose=True,\n"
            f")\n"
            f"\n"
            f"if not tiered:\n"
            f'    print("Error: no viable configurations found")\n'
            f"    sys.exit(1)\n"
            f"\n"
            f'print(f"Tiers found: {{list(tiered.keys())}}")\n'
            f"\n"
            f"tiers = json.loads({repr(tiers_json)})\n"
            f"paths = orch.generate_tiered_models(\n"
            f"    tiered=tiered, model_name_prefix={repr(model_name)}, "
            f"tiers=tiers, verify=False,\n"
            f")\n"
            f"valid = [p for p in paths if p]\n"
            f"for p in valid:\n"
            f"    size = os.path.getsize(p) / 1e9\n"
            f'    print(f"  {{Path(p).name}} ({{size:.1f}} GB)")\n'
            f'print(f"Generated {{len(valid)}} hybrid GGUF files")\n'
            f'print("PIPELINE_STAGE_COMPLETE=magicquant")\n'
        )
        return script


class UploadService:
    """Orchestrates the HuggingFace upload subprocess."""

    def __init__(self, pipeline_root: Path, venv_python: str) -> None:
        self.pipeline_root = pipeline_root
        self.venv_python = venv_python

    def build_script(
        self,
        *,
        repo_id: str,
        private: bool,
        license_id: str,
        upload_gguf: bool,
        upload_lora: bool,
        upload_merged: bool,
        upload_dataset: bool,
        base_model: str,
        dataset_name: str,
        did_training: bool = True,
        did_heretic: bool = False,
        did_magicquant: bool = True,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        num_epochs: int,
        learning_rate: float,
        max_seq_length: int,
        batch_size: int,
        gradient_accumulation: int,
        optimizer: str,
        lr_scheduler: str,
        out_abs: str,
    ) -> str:
        """Generate the upload subprocess script text."""
        script = (
            "import sys\n"
            'sys.path.insert(0, "core")\n'
            "from hf_upload import HFUploadConfig, upload\n"
            "\n"
            "cfg = HFUploadConfig(\n"
            f"    repo_id={repr(repo_id)},\n"
            f"    private={private},\n"
            f"    license={repr(license_id)},\n"
            f"    upload_gguf={upload_gguf},\n"
            f"    upload_lora={upload_lora},\n"
            f"    upload_merged={upload_merged},\n"
            f"    upload_dataset={upload_dataset},\n"
            f"    base_model={repr(base_model)},\n"
            f"    dataset_name={repr(dataset_name)},\n"
            f"    did_training={did_training},\n"
            f"    did_heretic={did_heretic},\n"
            f"    did_magicquant={did_magicquant},\n"
            f"    lora_r={lora_r},\n"
            f"    lora_alpha={lora_alpha},\n"
            f"    lora_dropout={lora_dropout},\n"
            f"    num_epochs={num_epochs},\n"
            f"    learning_rate={learning_rate},\n"
            f"    max_seq_length={max_seq_length},\n"
            f"    batch_size={batch_size},\n"
            f"    gradient_accumulation={gradient_accumulation},\n"
            f"    optimizer={repr(optimizer)},\n"
            f"    lr_scheduler={repr(lr_scheduler)},\n"
            f")\n"
            f"\n"
            f"ok = upload(cfg, {repr(out_abs)})\n"
            f"if ok:\n"
            f'    print("PIPELINE_STAGE_COMPLETE=upload")\n'
            f"else:\n"
            f"    sys.exit(1)\n"
        )
        return script
