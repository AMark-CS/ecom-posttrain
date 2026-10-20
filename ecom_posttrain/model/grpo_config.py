#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRPO config + pure prompt loader."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    model_path: str = ""
    data_path: str = ""
    output_dir: str = "./output/grpo"
    max_length: int = 768
    max_prompt_length: int = 320
    group_size: int = 4
    prompts_per_step: int = 2
    max_steps: int = 8
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    kl_beta: float = 0.02
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 96
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 1
    save_every: int = 100
    dtype: str = "bfloat16"
    ref_device: str = "cpu"
    use_wandb: bool = False
    wandb_project: str = "mini-posttrain"
    max_examples: int = -1
    reward_weights: dict[str, float] = field(default_factory=dict)

def load_prompts(data_path: str, max_examples: int = -1) -> list[dict[str, Any]]:
    """Load prompts + meta for reward scoring."""
    p = Path(data_path)
    raw: list[dict] = []
    if p.suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
    else:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    out = []
    for ex in raw:
        prompt = ex.get("prompt")
        if not prompt:
            continue
        meta = ex.get("meta", {}) or {}
        out.append({"prompt": prompt, "meta": meta})
    if max_examples > 0:
        out = out[:max_examples]
    return out
