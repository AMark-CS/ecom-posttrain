#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/model/dpo_train.py
====================================
偏好优化（DPO / RL）训练脚本（模块 3：偏好优化）

基于模块 2 的 SFT 模型，使用构造的 Preference 数据集（优劣对）进行 DPO 训练。
Key points:
  - 参考模型放在 CPU 节省显存（ref_device=cpu），适合低显存环境
  - beta=0.1 控制偏好强度
  - gamma=0.1 额外加入 SFT 正则项，防止分布偏移
  - Data: dataset/processed/preference_train.jsonl

Usage example:
    python model/dpo_train.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --data_path dataset/processed/preference_train.jsonl \
        --output_dir output/ecommerce_dpo \
        --use_lora --ref_device cpu --beta 0.1 --gamma 0.1 \
        --max_steps 50 --per_device_batch_size 2 --grad_accum_steps 8
"""

from __future__ import annotations
# --- console / import bootstrap ---
import sys
from pathlib import Path
try:
    from ecom_posttrain._console import bootstrap
    bootstrap()
except Exception:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nano_posttrain.cli.train_dpo import parse_args
from nano_posttrain.trainers import DPOTrainer
from nano_posttrain.trainers.dpo_trainer import DPOConfig


def parse_ecommerce_dpo_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="电商场景 DPO 偏好优化训练")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="output/ecommerce_dpo")
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--no_lora", dest="use_lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.1, help="DPO 偏好强度")
    p.add_argument("--gamma", type=float, default=0.1, help="SFT 正则权重")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--ref_device", type=str, default="cpu", choices=["cpu", "cuda"],
                   help="参考模型设备：cpu 节省 ~50% 显存")
    return p.parse_args()


def run_dpo(args: argparse.Namespace) -> None:
    fields = [f for f in DPOConfig.__dataclass_fields__]
    cfg_dict = {k: getattr(args, k) for k in vars(args) if k in fields}
    cfg = DPOConfig(**cfg_dict)
    trainer = DPOTrainer(cfg)
    trainer.fit()
    print(f"\n✅ 电商 DPO 训练完成。输出：{args.output_dir}")
    report_path = Path(args.output_dir) / "ecommerce_dpo_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "use_lora": args.use_lora,
            "beta": args.beta,
            "gamma": args.gamma,
            "ref_device": args.ref_device,
            "scenario": "ecommerce_preference_optimization",
            "tasks": ["product_description_quality", "customer_service_tone", "recommendation_accuracy"],
            "key_trick": "ref-on-CPU saves ~50% VRAM",
        }, f, ensure_ascii=False, indent=2)
    print(f"📊 报告已保存：{report_path}")


def main() -> int:
    args = parse_ecommerce_dpo_args()
    run_dpo(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
