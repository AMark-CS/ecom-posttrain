#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRPO config + pure prompt loader."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import math

@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    model_path: str = ""
    data_path: str = ""
    output_dir: str = "./output/grpo"
    max_length: int = 768
    max_prompt_length: int = 320
    clip_epsilon: float = .2
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
    wandb_project: str = "ecom-posttrain"
    max_examples: int = -1
    reward_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("learning_rate", "weight_decay", "warmup_ratio", "kl_beta", "temperature", "top_p", "lora_alpha", "lora_dropout", "grad_clip", "clip_epsilon"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.warmup_ratio <= 1 or self.weight_decay < 0 or not 0 <= self.lora_dropout < 1 or self.grad_clip <= 0:
            raise ValueError("Invalid optimizer or dropout settings")
        if self.group_size < 2 or self.prompts_per_step < 1 or self.max_steps < 1:
            raise ValueError("GRPO needs at least two completions, one prompt and one step")
        if self.temperature <= 0 or not 0 < self.top_p <= 1 or self.kl_beta < 0 or not 0 < self.clip_epsilon < 1:
            raise ValueError("Invalid sampling or objective settings")
        if self.max_new_tokens < 1 or self.max_prompt_length < 1 or self.max_prompt_length + self.max_new_tokens > self.max_length:
            raise ValueError("Prompt and completion budgets must fit max_length")
        if self.learning_rate <= 0 or self.save_every < 1 or self.log_every < 1:
            raise ValueError("Invalid optimizer/checkpoint settings")
        if self.dtype not in {"bfloat16", "float16", "float32"} or self.ref_device not in {"cpu", "cuda"}:
            raise ValueError("Unsupported dtype or reference device")

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
        meta = {**meta.get("tags", {}), **meta}
        out.append({"prompt": prompt, "meta": meta})
    if max_examples > 0:
        out = out[:max_examples]
    return out
