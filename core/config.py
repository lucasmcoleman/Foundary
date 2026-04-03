"""
Centralized Foundry configuration using pydantic-settings.

All settings can be overridden via:
  1. Constructor keyword arguments
  2. Environment variables with FOUNDRY_ prefix (e.g. FOUNDRY_OUTPUT_DIR)
  3. A .env file in the project root

Usage:
    from core.config import settings
    print(settings.model_name)
    print(settings.output_dir)
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FoundrySettings(BaseSettings):
    """Foundry-wide configuration loaded from env vars and .env file."""

    model_config = SettingsConfigDict(
        env_prefix="FOUNDRY_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # -- Paths --
    output_dir: Path = Path("./output")
    llamacpp_path: Optional[Path] = None

    # -- Training --
    model_name: str = "Tesslate/OmniCoder-9B"
    dataset_path: Path = Path("data/zeroclaw_training_data.jsonl")
    max_seq_length: int = 8192
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    optim: str = "adamw_8bit"

    # -- Export --
    gguf_type: str = "bf16"

    # -- Heretic (abliteration) --
    heretic_n_trials: int = 200
    heretic_quantization: str = "bnb_4bit"
    heretic_kl_scale: float = 1.0

    # -- MagicQuant --
    target_base_quant: str = "MXFP4_MOE"
    mq_generations: int = 50
    mq_population_size: int = 100
    mq_tiers: list[str] = Field(default=["Q4", "Q5", "Q6"])

    # -- Upload --
    hf_repo_id: str = ""
    hf_private: bool = True
    hf_license: str = "apache-2.0"

    # -- UI --
    ui_port: int = 7865
    api_key: str = ""  # Set FOUNDRY_API_KEY env var to enable auth

    # -- ROCm --
    device: str = "cuda:0"
    hsa_enable_sdma: str = "0"
    pytorch_hip_alloc_conf: str = "backend:native,expandable_segments:True"

    # -- HuggingFace --
    hf_token: Optional[str] = None


# Module-level singleton for convenience imports.
# Importing `settings` triggers env var / .env resolution once.
settings = FoundrySettings()
