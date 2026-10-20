#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/model/sft_train.py
====================================
监督微调（SFT）训练脚本（模块 2）

集成 mini_posttrain 的 SFTTrainer，针对电商数据（商品描述生成、智能客服、推荐解释）进行微调。
支持 QLoRA（低显存）和全参数微调选项。

Key points:
  - 使用 LoRA / QLoRA（r=16, alpha=32, dropout=0.05）
  - 学习率 5e-5，batch_size=4，grad_accum=4
  - 混合精度 bfloat16
  - 数据来源：dataset/processed/sft_train.jsonl

Usage example:
    python model/sft_train.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --data_path dataset/processed/sft_train.jsonl \
        --output_dir output/ecommerce_sft \
        --use_lora --lora_r 16 --lora_alpha 32 \
        --max_steps 100 --per_device_batch_size 2 --grad_accum_steps 4 \
        --learning_rate 5e-5 --num_epochs 1
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

# 复用 mini_posttrain 核心库
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nano_posttrain.cli.train_sft import parse_args, main as cli_main
from nano_posttrain.trainers import SFTTrainer
from nano_posttrain.trainers.sft_trainer import SFTConfig


def parse_ecommerce_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="电商场景 SFT 微调（商品描述生成 / 客服 / 推荐解释）"
    )
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                   help="基础模型（建议 Qwen2.5 系列支持多模态）")
    p.add_argument("--data_path", type=str, required=True,
                   help="SFT 数据文件（JSONL 格式，每行含 messages）")
    p.add_argument("--output_dir", type=str, default="output/ecommerce_sft",
                   help="模型权重与检查点保存路径")
    p.add_argument("--use_lora", action="store_true", default=True,
                   help="启用 LoRA 微调（默认）")
    p.add_argument("--no_lora", dest="use_lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA 秩")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA 缩放系数")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--grad_checkpointing", action="store_true")
    return p.parse_args()


def run_sft(args: argparse.Namespace) -> None:
    # 转换为 SFTConfig
    fields = [f for f in SFTConfig.__dataclass_fields__]
    cfg_dict = {k: getattr(args, k) for k in vars(args) if k in fields}
    cfg = SFTConfig(**cfg_dict)
    trainer = SFTTrainer(cfg)
    trainer.fit()
    print(f"\n✅ 电商 SFT 训练完成。输出：{args.output_dir}")
    # 生成简要报告
    report_path = Path(args.output_dir) / "ecommerce_sft_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "use_lora": args.use_lora,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "data_path": args.data_path,
            "max_steps": args.max_steps,
            "output_dir": args.output_dir,
            "scenario": "ecommerce_search_recommend_content",
            "tasks": ["product_description", "customer_service", "recommendation_explanation"],
        }, f, ensure_ascii=False, indent=2)
    print(f"📊 报告已保存：{report_path}")


def main() -> int:
    args = parse_ecommerce_args()
    run_sft(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
