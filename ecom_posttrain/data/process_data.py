#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/data/process_data.py
=====================================
电商数据清洗与标注脚本（模块 1：数据清洗与标注）

职责：
  1. 读取原始数据（商品 / 行为）
  2. 去重、去噪、异常值处理
  3. 统一格式 -> 统一 JSON schema
  4. 质量检查（长度、字段完整性、注入检测）
  5. 业务标签生成（价格带、材质、类别、热度、多语言）
  6. 数据集切分 train / valid / test
  7. 输出到 dataset/processed/

用法:
    python process_data.py --input_dir dataset --output_dir dataset/processed --valid_ratio 0.1
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
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

random.seed(42)

# 质量检查阈值
MIN_DESC_LEN = 20
MAX_DESC_LEN = 2000
REQUIRED_PRODUCT_FIELDS = ["product_id", "title", "description", "price", "category"]


# ---------------------------------------------------------------------------
# 质量检查
# ---------------------------------------------------------------------------

def quality_check_product(p: dict[str, Any]) -> tuple[bool, list[str]]:
    """返回 (是否通过, 问题列表)。"""
    issues: list[str] = []
    for field in REQUIRED_PRODUCT_FIELDS:
        if field not in p or p[field] in (None, ""):
            issues.append(f"missing_field:{field}")
    desc = p.get("description", "")
    if isinstance(desc, str):
        if len(desc) < MIN_DESC_LEN:
            issues.append("desc_too_short")
        if len(desc) > MAX_DESC_LEN:
            issues.append("desc_too_long")
        if re.search(r"(ignore previous|忽略以上|disregard all)", desc, re.IGNORECASE):
            issues.append("prompt_injection")
    price = p.get("price")
    if not isinstance(price, (int, float)) or price <= 0:
        issues.append("invalid_price")
    return (len(issues) == 0), issues


# ---------------------------------------------------------------------------
# 业务标签
# ---------------------------------------------------------------------------

def price_band(price: float) -> str:
    if price < 50:
        return "budget"
    if price < 200:
        return "mid_low"
    if price < 1000:
        return "mid_high"
    return "premium"


def sales_band(sales: int) -> str:
    if sales < 100:
        return "niche"
    if sales < 5000:
        return "normal"
    if sales < 20000:
        return "hot"
    return "bestseller"


def rating_band(rating: float) -> str:
    if rating < 4.0:
        return "poor"
    if rating < 4.5:
        return "average"
    if rating < 4.8:
        return "good"
    return "excellent"


def add_tags(p: dict[str, Any]) -> dict[str, Any]:
    """附加业务标签。"""
    p = dict(p)
    p["tags"] = {
        "price_band": price_band(float(p.get("price", 0))),
        "sales_band": sales_band(int(p.get("sales_volume", 0))),
        "rating_band": rating_band(float(p.get("rating", 0))),
        "category": p.get("category", "unknown"),
        "material": p.get("material", "unknown"),
        "language": p.get("language", "zh"),
        "is_multi_modal": bool(p.get("image_url")),
    }
    return p


# ---------------------------------------------------------------------------
# 统一 schema
# ---------------------------------------------------------------------------

SFT_SCHEMA = {
    "messages": "list[dict(role,content)] — 标准 chat 格式",
    "meta": "dict — 业务标签与来源追踪",
}


def to_unified_sft(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """商品数据 -> 统一 SFT schema。"""
    unified = []
    for p in products:
        is_zh = p.get("language", "zh") == "zh"
        tasks = [
            (
                "请根据以下商品信息生成一段吸引人的商品描述。"
                if is_zh else
                "Please write an attractive product description based on the following product information.",
                (
                    f"品牌：{p['brand']}\n类别：{p['category']}\n材质：{p['material']}\n"
                    f"价格：{p['price']} {p['currency']}\n"
                ) if is_zh else
                (
                    f"Brand: {p['brand']}\nCategory: {p['category']}\nMaterial: {p['material']}\n"
                    f"Price: {p['price']} {p['currency']}\n"
                ),
                p["description"],
                "product_description",
            ),
            (
                f"这款{p['title']}多少钱？" if is_zh else f"How much does {p['title']} cost?",
                "",
                f"{p['title']}的价格是{p['price']}{p['currency']}。" if is_zh
                else f"The price of {p['title']} is {p['price']} {p['currency']}.",
                "customer_service",
            ),
            (
                f"为什么向你推荐这款{p['category']}？" if is_zh
                else f"Why do you recommend this {p['category']} to me?",
                "",
                (
                    f"这款{p['brand']}的{p['category']}采用{p['material']}材质，售价{p['price']}{p['currency']}，"
                    f"已售出{p['sales_volume']}件，用户评分{p['rating']}分，是同价位中口碑优秀的选择。"
                ) if is_zh else
                (
                    f"This {p['brand']} {p['category']} is made of {p['material']}, priced at "
                    f"{p['price']} {p['currency']}, with {p['sales_volume']} units sold and a "
                    f"{p['rating']} star rating - an excellent choice in its price range."
                ),
                "recommendation_explanation",
            ),
        ]
        for instruction, user_input, output, task_type in tasks:
            content = instruction + ("\n" + user_input if user_input else "")
            unified.append(
                {
                    "messages": [
                        {"role": "user", "content": content},
                        {"role": "assistant", "content": output},
                    ],
                    "meta": {
                        "task_type": task_type,
                        "product_id": p["product_id"],
                        "language": p.get("language", "zh"),
                        "tags": p.get("tags", {}),
                        "source": "ecommerce_mock",
                    },
                }
            )
    return unified


def to_unified_preference(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """商品数据 -> 统一 Preference schema。"""
    unified = []
    for p in products:
        is_zh = p.get("language", "zh") == "zh"
        prompt = (
            f"请根据以下商品信息生成一段吸引人的商品描述。\n"
            f"品牌：{p['brand']}\n类别：{p['category']}\n材质：{p['material']}\n价格：{p['price']} {p['currency']}"
        ) if is_zh else (
            f"Please write an attractive product description.\n"
            f"Brand: {p['brand']}\nCategory: {p['category']}\nMaterial: {p['material']}\nPrice: {p['price']} {p['currency']}"
        )
        chosen = p["description"]
        # 低质量版本：信息缺失 + 冗余
        rejected = (
            f"这个商品挺好的，就是一个普通的东西。价格 {p['price']}，喜欢就买吧。。。"
        ) if is_zh else (
            f"It's a nice product, just a normal thing. Price {p['price']}, buy if you like it..."
        )
        unified.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "meta": {
                    "product_id": p["product_id"],
                    "task_type": "product_description",
                    "language": p.get("language", "zh"),
                    "preference_source": "rule_based_quality",
                    "brand": p.get("brand", ""),
                    "category": p.get("category", ""),
                    "material": p.get("material", ""),
                    "price": p.get("price", 0),
                    "price_band": p.get("tags", {}).get("price_band", ""),
                    "tags": p.get("tags", {}),
                },
            }
        )
    return unified


def to_multimodal(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """商品数据 -> 统一多模态 schema（图像 + 文本）。

    采用早期融合的输入格式 ``<image>`` + 文本，与 Qwen2.5-VL 系列接口一致。
    """
    unified = []
    for p in products:
        is_zh = p.get("language", "zh") == "zh"
        instruction = "请描述这张商品图片。" if is_zh else "Please describe this product image."
        unified.append(
            {
                "image": p.get("image_url", ""),
                "messages": [
                    {"role": "user", "content": f"<image>\n{instruction}"},
                    {"role": "assistant", "content": p["description"]},
                ],
                "meta": {
                    "product_id": p["product_id"],
                    "task_type": "multimodal_product_description",
                    "language": p.get("language", "zh"),
                    "tags": p.get("tags", {}),
                    "fusion_strategy": "early_fusion",
                },
            }
        )
    return unified


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------

def split_dataset(data: list[Any], valid_ratio: float, test_ratio: float = 0.0) -> dict[str, list[Any]]:
    """按比例切分并打乱。"""
    data = list(data)
    random.shuffle(data)
    n = len(data)
    n_valid = int(n * valid_ratio)
    n_test = int(n * test_ratio)
    return {
        "train": data[: n - n_valid - n_test],
        "valid": data[n - n_valid - n_test: n - n_test],
        "test": data[n - n_test:],
    }


def save_jsonl(data: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(data: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="电商数据清洗与标注脚本")
    parser.add_argument("--input_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="dataset/processed")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 载入商品数据 ----
    clean_path = in_dir / "products_clean.json"
    if clean_path.exists():
        with open(clean_path, encoding="utf-8") as f:
            products = json.load(f)
    else:
        raw = []
        for p in sorted(in_dir.glob("products_raw_*.json")):
            with open(p, encoding="utf-8") as f:
                raw.extend(json.load(f))
        products = clean_products_fallback(raw)
    print(f"[1/5] 载入商品数据：{len(products)} 条")

    # ---- 质量检查 ----
    print("[2/5] 质量检查中...")
    passed, rejected, issue_counter = [], [], Counter()
    for p in products:
        ok, issues = quality_check_product(p)
        if ok:
            passed.append(p)
        else:
            rejected.append(p)
            issue_counter.update(issues)
    print(f"       通过：{len(passed)} | 剔除：{len(rejected)}")
    if issue_counter:
        print(f"       问题分布：{dict(issue_counter)}")

    # ---- 标注 ----
    print("[3/5] 生成业务标签...")
    tagged = [add_tags(p) for p in passed]
    save_json(tagged, out_dir / "products_tagged.json")

    # ---- 统一格式 + 切分 ----
    print("[4/5] 统一格式转换与数据集切分...")
    sft_unified = to_unified_sft(tagged)
    sft_splits = split_dataset(sft_unified, args.valid_ratio, args.test_ratio)
    for name, split in sft_splits.items():
        save_jsonl(split, out_dir / f"sft_{name}.jsonl")
    print(f"       SFT: train={len(sft_splits['train'])} valid={len(sft_splits['valid'])} test={len(sft_splits['test'])}")

    pref_unified = to_unified_preference(tagged)
    pref_splits = split_dataset(pref_unified, args.valid_ratio, args.test_ratio)
    for name, split in pref_splits.items():
        save_jsonl(split, out_dir / f"preference_{name}.jsonl")
    print(f"       Pref: train={len(pref_splits['train'])} valid={len(pref_splits['valid'])} test={len(pref_splits['test'])}")

    mm_unified = to_multimodal(tagged)
    mm_splits = split_dataset(mm_unified, args.valid_ratio, args.test_ratio)
    for name, split in mm_splits.items():
        save_jsonl(split, out_dir / f"multimodal_{name}.jsonl")
    print(f"       MM:   train={len(mm_splits['train'])} valid={len(mm_splits['valid'])} test={len(mm_splits['test'])}")

    # ---- 统计报告 ----
    print("[5/5] 生成数据统计报告...")
    lang_dist = Counter(p["language"] for p in tagged)
    cat_dist = Counter(p["category"] for p in tagged)
    report = {
        "total_products": len(tagged),
        "rejected_products": len(rejected),
        "rejection_reasons": dict(issue_counter),
        "language_distribution": dict(lang_dist),
        "category_distribution": dict(cat_dist),
        "sft_examples": {k: len(v) for k, v in sft_splits.items()},
        "preference_examples": {k: len(v) for k, v in pref_splits.items()},
        "task_distribution": dict(
            Counter(ex["meta"]["task_type"] for ex in sft_unified)
        ),
    }
    with open(out_dir / "data_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"\n✅ 数据处理完成 -> {out_dir.resolve()}")


def clean_products_fallback(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """在没有 products_clean.json 时，直接从 raw 做一次清洗。"""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from prepare_data import clean_products  # type: ignore

    return clean_products(raw)


if __name__ == "__main__":
    main()
