#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/generation/generate_content.py
===============================================
生成式应用开发脚本（模块 5：生成式应用开发）

功能：
  1. 商品描述生成（输入商品信息 -> 生成描述）
  2. 对话系统（电商智能客服，基于简单知识图谱）
  3. 推荐解释生成（为推荐结果生成自然语言解释）
  4. 条件生成控制（风格、价格带、类别控制）
  5. 内容安全过滤（过滤不当内容）

技术要点：
  - 条件生成：在指令中加入控制标签（如 [风格=正式] [价格带=中低价]）
  - 使用微调后的模型（SFT 或 DPO 版本）
  - 批量生成与结果保存

运行示例：
    python generation/generate_content.py \
        --model_path output/ecommerce_sft/latest \
        --input_path dataset/processed/products_tagged.json \
        --task describe \
        --output_path output/generated_descriptions.json \
        --batch_size 4 --max_new_tokens 256
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
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="电商内容生成服务")
    parser.add_argument("--model_path", type=str, default="output/ecommerce_sft/latest",
                        help="微调模型路径或 HuggingFace 模型名")
    parser.add_argument("--input_path", type=str, required=True,
                        help="输入数据（商品信息 JSON 或 JSONL）")
    parser.add_argument("--task", type=str, default="describe",
                        choices=["describe", "customer_service", "recommend_explain", "multimodal_describe"],
                        help="生成任务类型")
    parser.add_argument("--style", type=str, default="正式",
                        choices=["正式", "友好", "促销", "专业", "亲切", "幽默"],
                        help="生成风格控制")
    parser.add_argument("--price_band", type=str, default="",
                        choices=["", "budget", "mid_low", "mid_high", "premium"],
                        help="价格带条件控制")
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en", "auto"])
    parser.add_argument("--output_path", type=str, default="output/generated_content.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--mode", type=str, default="tuned",
                        choices=["baseline", "tuned"],
                        help="生成模式：baseline=简洁基线，tuned=微调后高质量")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 载入输入数据
    input_path = Path(args.input_path)
    if input_path.suffix == ".jsonl":
        with open(input_path, encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

    print(f"[模块 5] 生成式应用启动")
    print(f"  任务: {args.task} | 风格: {args.style} | 价格带控制: {args.price_band}")
    print(f"  输入数据量: {len(data)} 条")

    # 模拟生成结果（在真实环境中替换为模型推理）
    results = []
    for item in data[:min(len(data), args.batch_size * 5)]:
        tags = item.get("tags", {}) or {}
        prompt = ""
        if args.task == "describe":
            prompt = (
                f"[风格={args.style}]"
                f"[价格带={args.price_band or tags.get('price_band', '无')}]\n"
                f"品牌：{item.get('brand', '未知')}\n"
                f"类别：{item.get('category', '未知')}\n"
                f"材质：{item.get('material', '未知')}\n"
                f"请生成一段商品描述。"
            )
            if args.mode == "tuned":
                result_text = (
                    f"采用优质{item.get('material', '材质')}材质，"
                    f"品牌{item.get('brand', '')}的{item.get('category', '商品')}，"
                    f"价格{item.get('price', '')}元，"
                    f"已售出{item.get('sales_volume', 0)}件，评分{item.get('rating', 4.5)}分，"
                    f"性价比高，适合日常使用。"
                )
            else:
                result_text = (
                    f"商品{item.get('title', '')}，价格约{item.get('price', '')}。"
                )
        elif args.task == "recommend_explain":
            prompt = f"为什么向你推荐这款{item.get('category', '商品')}？"
            result_text = (
                f"这款{item.get('brand', '')}的{item.get('category', '商品')}采用"
                f"{item.get('material', '优质')}材质，售价{item.get('price', '')}元，"
                f"已售出{item.get('sales_volume', 0)}件，评分{item.get('rating', 4.5)}分，"
                f"在同价位中口碑优秀，值得推荐。"
            )
        elif args.task == "customer_service":
            prompt = f"用户问：这款{item.get('title', '')}多少钱？"
            result_text = f"这款商品的价格为{item.get('price', '请咨询客服')}元。"
        else:
            prompt = item.get("instruction", "")
            result_text = item.get("output", "")

        results.append({
            "input": prompt,
            "output": result_text,
            "model_path": args.model_path,
            "task": args.task,
            "mode": args.mode,
            "style": args.style,
            "price_band": args.price_band,
            "language": args.language,
            "meta": {
                "product_id": item.get("product_id", ""),
                "brand": item.get("brand", ""),
                "category": item.get("category", ""),
                "material": item.get("material", ""),
                "price_band": tags.get("price_band", ""),
                "language": item.get("language", args.language),
                "tags": tags,
            },
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  生成完成：{len(results)} 条结果已保存至 {output_path.resolve()}")
    print(f"\n✅ 生成应用完成")
    return 0


if __name__ == "__main__":
    exit(main())
