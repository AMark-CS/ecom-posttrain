#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/generation/post_process.py
=============================================
内容后处理脚本（模块 5：后处理）

功能：
  1. 格式化（JSON 规范化、字段重命名、缺失值填充）
  2. 多语言翻译（中 -> 英 或 英 -> 中）
  3. 质量检查（重复率、关键词覆盖率、可读性评分）
  4. 批量导出（CSV / HTML / Markdown）

运行：
    python generation/post_process.py \
        --input_path output/generated_content_filtered.json \
        --format json \
        --quality_check --output_path output/final_content.json
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


def normalize_content(data: list[dict[str, object]]) -> list[dict[str, object]]:
    for item in data:
        meta = item.get("meta", {}) or {}
        filter_result = item.get("filter_result", {}) or {}
        # 统一格式
        item["final_text"] = item.get("output", "") if (filter_result.get("passed", True)) else "[已过滤]内容不符合安全规范"
        item["formatted_meta"] = {
            "product_id": meta.get("product_id", ""),
            "category": meta.get("tags", {}).get("category", "unknown"),
            "price_band": meta.get("tags", {}).get("price_band", "unknown"),
            "language": meta.get("language", "zh"),
            "filter_level": filter_result.get("level", "unknown"),
        }
    return data


def quality_score(item: dict[str, object]) -> dict[str, float]:
    text = str(item.get("final_text", ""))
    score = 0.0
    # 关键词覆盖
    keywords = ["品牌", "材质", "价格", "Brand", "Material", "Price"]
    hits = sum(1 for k in keywords if k in text)
    score += min(hits / len(keywords), 1.0) * 0.5
    # 长度合理性
    length = len(text)
    score += (1 - abs(length - 150) / 300) if 0 < length < 600 else 0.0
    # 重复惩罚
    if text.count("。") > len(text) / 10:
        score -= 0.2
    return {"score": max(0.0, score), "keyword_coverage": hits / len(keywords)}


def main() -> int:
    parser = argparse.ArgumentParser(description="内容后处理")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--format", type=str, default="json", choices=["json", "csv", "markdown"])
    parser.add_argument("--quality_check", action="store_true", default=True)
    parser.add_argument("--output_path", type=str, default="output/final_content.json")
    args = parser.parse_args()

    with open(args.input_path, encoding="utf-8") as f:
        data = json.load(f)

    data = normalize_content(data)
    for item in data:
        if args.quality_check:
            item["quality_score"] = quality_score(item)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 后处理完成：{len(data)} 条内容已保存至 {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    exit(main())
