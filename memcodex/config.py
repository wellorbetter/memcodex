"""Configuration for memcodex.

Environment variable overrides (MEMCODEX_ prefix):
  MEMCODEX_LLM_BASE_URL, MEMCODEX_LLM_API_KEY, MEMCODEX_LLM_MODEL,
  MEMCODEX_DATASET_PATH, MEMCODEX_TOTAL_BUDGET
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    llm_base_url: str = "http://127.0.0.1:18200/v1"
    llm_api_key: str = "local"
    llm_model: str = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
    dataset_path: str = "examples/demo_dataset/episodes.json"
    total_budget: int = 1024

    @classmethod
    def from_env(cls, base: "Config | None" = None) -> "Config":
        cfg = base or cls()
        env = {
            "MEMCODEX_LLM_BASE_URL": "llm_base_url",
            "MEMCODEX_LLM_API_KEY": "llm_api_key",
            "MEMCODEX_LLM_MODEL": "llm_model",
            "MEMCODEX_DATASET_PATH": "dataset_path",
            "MEMCODEX_TOTAL_BUDGET": "total_budget",
        }
        for env_key, field in env.items():
            val = os.environ.get(env_key)
            if val is not None:
                setattr(cfg, field, int(val) if field == "total_budget" else val)
        return cfg


def load_config() -> Config:
    return Config.from_env()