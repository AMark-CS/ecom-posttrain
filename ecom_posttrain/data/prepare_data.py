#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/data/prepare_data.py
======================================
电商数据采集与构建脚本（模块 1：数据构建）

本脚本模拟电商场景的全量数据采集：
  1. 商品信息采集（标题 / 描述 / 图片 / 价格 / 分类 / 销量）
  2. 用户行为数据（点击、收藏、购买记录、搜索日志）
  3. 多语言数据（中 / 英商品数据）
  4. 数据清洗与去重
  5. 统一格式转换为 JSON/JSONL
  6. 业务标签生成（价格带、型号、材质等）
  7. SFT 指令-响应对与 Preference 优劣对构造

输出目录：dataset/
  - products_raw.json          原始商品数据
  - products_clean.json        清洗后商品数据
  - user_behavior.json         用户行为数据
  - sft_dataset.json           SFT 训练集（instruction / input / output）
  - preference_dataset.json    DPO 偏好数据集（prompt / chosen / rejected）
  - multimodal_dataset.json    多模态数据集（image + text）

运行方式：
    python prepare_data.py --output_dir dataset --num_products 2000 --lang zh,en
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
from pathlib import Path
from typing import Any

# 固定随机种子，保证可复现
random.seed(42)

# ---------------------------------------------------------------------------
# 1. 模拟数据源（在实际生产环境中，这些将来自数据库 / 数据湖 / API）
# ---------------------------------------------------------------------------

PRODUCT_CATEGORIES_ZH = [
    "电子产品", "家居家纺", "服装穿搭", "美妆护肤", "食品生鲜",
    "母婴儿童", "运动户外", "图书音像", "汽车用品", "宠物用品",
]
PRODUCT_CATEGORIES_EN = [
    "Electronics", "Home & Textiles", "Fashion & Apparel", "Beauty & Skincare",
    "Food & Fresh", "Baby & Kids", "Sports & Outdoors", "Books & Media",
    "Automotive", "Pet Supplies",
]

BRANDS_ZH = ["小米", "华为", "安克", "李宁", "完美日记", "蒙牛", "波司登", "蓝猫", "得力", "九阳"]
BRANDS_EN = ["Xiaomi", "Huawei", "Anker", "Li-Ning", "Perfect Diary", "Mengniu", "Bosideng", "BlueCat", "Deli", "Joyoung"]

MATERIALS = ["棉", "涤纶", "真皮", "不锈钢", "塑料", "木质", "硅胶", "玻璃", "铝合金", "羊毛"]

STYLE_OPTIONS = ["正式", "友好", "促销", "专业", "亲切", "幽默"]

LANGUAGES = ["zh", "en"]

DESCRIPTION_TEMPLATES_ZH = [
    "这款{category}由{brand}推出，采用{material}材质，适合{scenarios}场景使用。",
    "{brand}的这款{category}，主打{material}工艺，{price_desc}，{feature_desc}。",
    "如果你在找{category}，这款{brand}产品不容错过：{material}材质，{selling_point}，{usage_scenario}。",
]
DESCRIPTION_TEMPLATES_EN = [
    "This {category} by {brand} is made of {material}, suitable for {scenarios}.",
    "{brand}'s {category} features {material} craftsmanship, {price_desc}. {feature_desc}.",
    "Looking for a {category}? This {brand} product is a great choice: {material}, {selling_point}, perfect for {usage_scenario}.",
]

SEARCH_QUERIES_ZH = [
    "夏季短袖T恤", "无线蓝牙耳机", "保湿面霜", "有机牛奶", "儿童玩具",
    "跑步鞋", "笔记本电脑", "猫粮", "护肤品套装", "厨房刀具",
]
SEARCH_QUERIES_EN = [
    "summer t-shirt", "wireless bluetooth earphones", "moisturizing cream", "organic milk",
    "children toys", "running shoes", "laptop", "cat food", "skincare set", "kitchen knives",
]


def _random_price() -> float:
    return round(random.uniform(9.9, 5999.0), 2)


def _price_band(price: float) -> str:
    if price < 50:
        return "低价 (<¥50)"
    if price < 200:
        return "中低价 (¥50-¥200)"
    if price < 1000:
        return "中高价 (¥200-¥1000)"
    return "高价 (>¥1000)"


def _random_image_url(index: int, lang: str) -> str:
    """模拟商品图片 URL"""
    return f"https://cdn.example.com/products/{lang}/product_{index:05d}.jpg"


def generate_products(num_products: int, lang: str) -> list[dict[str, Any]]:
    """生成模拟商品数据。"""
    products = []
    categories = PRODUCT_CATEGORIES_ZH if lang == "zh" else PRODUCT_CATEGORIES_EN
    brands = BRANDS_ZH if lang == "zh" else BRANDS_EN
    templates = DESCRIPTION_TEMPLATES_ZH if lang == "zh" else DESCRIPTION_TEMPLATES_EN

    for i in range(num_products):
        category = random.choice(categories)
        brand = random.choice(brands)
        price = _random_price()
        material = random.choice(MATERIALS)
        template = random.choice(templates)
        description = template.format(
            category=category,
            brand=brand,
            material=material,
            price_desc=_price_band(price),
            feature_desc="支持多种使用场景" if lang == "zh" else "supports multiple usage scenarios",
            selling_point="性价比高，口碑优秀" if lang == "zh" else "high cost-performance, excellent reputation",
            usage_scenario="日常通勤 / 休闲出行" if lang == "zh" else "daily commute / casual outing",
            scenarios="日常生活" if lang == "zh" else "daily life",
        )
        products.append(
            {
                "product_id": f"P{lang.upper()}{i+1:06d}",
                "title": f"{brand} {category} - {material} 系列" if lang == "zh" else f"{brand} {category} {material} Series",
                "description": description,
                "brand": brand,
                "category": category,
                "price": price,
                "currency": "CNY" if lang == "zh" else "USD",
                "material": material,
                "image_url": _random_image_url(i + 1, lang),
                "sales_volume": random.randint(0, 50000),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "language": lang,
                "stock": random.randint(0, 10000),
            }
        )
    return products


def generate_user_behavior(products: list[dict[str, Any]], num_records: int) -> list[dict[str, Any]]:
    """生成模拟用户行为数据（点击 / 收藏 / 购买 / 搜索）。"""
    behaviors = []
    action_types = ["click", "favorite", "purchase", "search"]
    for i in range(num_records):
        product = random.choice(products)
        action = random.choice(action_types)
        behaviors.append(
            {
                "user_id": f"U{random.randint(1, 5000):06d}",
                "product_id": product["product_id"],
                "action": action,
                "timestamp": f"2024-0{random.randint(1,9)}-{random.randint(10,28):02d} "
                             f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
                "session_id": f"S{random.randint(1, 20000):06d}",
                "device": random.choice(["mobile", "pc", "tablet"]),
                "language": product["language"],
            }
        )
    return behaviors


# ---------------------------------------------------------------------------
# 2. 清洗与去重
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """基础文本清洗：去除多余空白、控制字符、HTML 标签残留。"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)  # 去 HTML 标签
    text = re.sub(r"\s+", " ", text).strip()  # 合并空白
    return text


def clean_products(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """商品数据清洗：去重、字段标准化、缺失值处理。"""
    seen: set[str] = set()
    cleaned = []
    for p in products:
        title = clean_text(p.get("title", ""))
        desc = clean_text(p.get("description", ""))
        if not title or not desc:
            continue
        key = f"{p.get('brand','')}|{title}|{p.get('category','')}"
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "product_id": p["product_id"],
                "title": title,
                "description": desc,
                "brand": p.get("brand", "未知"),
                "category": p.get("category", "未知分类"),
                "price": float(p.get("price", 0)),
                "currency": p.get("currency", "CNY"),
                "material": p.get("material", "未知"),
                "image_url": p.get("image_url", ""),
                "sales_volume": int(p.get("sales_volume", 0)),
                "rating": float(p.get("rating", 0)),
                "language": p.get("language", "zh"),
                "stock": int(p.get("stock", 0)),
                "price_band": _price_band(float(p.get("price", 0))),
            }
        )
    return cleaned


# ---------------------------------------------------------------------------
# 3. SFT 数据构建（指令 - 响应对）
# ---------------------------------------------------------------------------

def build_sft_dataset(products: list[dict[str, Any]]) -> list[dict[str, str]]:
    """构建电商领域 SFT 数据：商品描述生成 / 智能客服 / 推荐解释。"""
    sft = []
    for p in products:
        lang = p["language"]
        is_zh = lang == "zh"

        # 任务 1：商品描述生成
        instr_zh = "请根据以下商品信息生成一段吸引人的商品描述。"
        instr_en = "Please write an attractive product description based on the following product information."
        input_text = (
            f"品牌：{p['brand']}\n类别：{p['category']}\n材质：{p['material']}\n"
            f"价格：{p['price']} {p['currency']}\n销量：{p['sales_volume']}\n"
        ) if is_zh else (
            f"Brand: {p['brand']}\nCategory: {p['category']}\nMaterial: {p['material']}\n"
            f"Price: {p['price']} {p['currency']}\nSales: {p['sales_volume']}\n"
        )
        output_text = p["description"]
        sft.append({"instruction": instr_zh if is_zh else instr_en, "input": input_text, "output": output_text})

        # 任务 2：智能客服 - 价格咨询
        q_zh = f"这款{p['title']}多少钱？"
        q_en = f"How much does {p['title']} cost?"
        a_zh = f"{p['title']}的价格是{p['price']}{p['currency']}。"
        a_en = f"The price of {p['title']} is {p['price']} {p['currency']}."
        sft.append({"instruction": q_zh if is_zh else q_en, "input": "", "output": a_zh if is_zh else a_en})

        # 任务 3：推荐解释
        r_zh = f"为什么向你推荐这款{p['category']}？"
        r_en = f"Why do you recommend this {p['category']} to me?"
        expl_zh = (
            f"这款{p['brand']}的{p['category']}采用{p['material']}材质，"
            f"价格{p['price']}{p['currency']}，销量达{p['sales_volume']}件，用户评分{p['rating']}分，"
            f"性价比高，非常适合日常使用。"
        )
        expl_en = (
            f"This {p['brand']} {p['category']} is made of {p['material']}, priced at "
            f"{p['price']} {p['currency']}, with {p['sales_volume']} units sold and a "
            f"{p['rating']} star rating. High cost-performance, great for daily use."
        )
        sft.append({"instruction": r_zh if is_zh else r_en, "input": "", "output": expl_zh if is_zh else expl_en})

    return sft


# ---------------------------------------------------------------------------
# 4. Preference 数据构造（优劣对）
# ---------------------------------------------------------------------------

def _quality_score(text: str) -> float:
    """简单的质量打分代理：长度、是否包含关键信息、是否有重复。"""
    if not text:
        return 0.0
    score = 0.0
    score += min(len(text) / 200, 1.0) * 0.4  # 长度适中加分
    if any(k in text for k in ("品牌", "材质", "价格", "Brand", "Material", "Price")):
        score += 0.3
    if "。。" in text or ".." in text:
        score -= 0.2  # 重复标点扣分
    return max(0.0, min(1.0, score))


def build_preference_dataset(sft_data: list[dict[str, str]]) -> list[dict[str, str]]:
    """基于 SFT 生成结果构造优劣对：高质量回答 vs 低质量回答。"""
    preference = []
    for ex in sft_data:
        prompt = ex["instruction"] + ("\n" + ex["input"] if ex["input"] else "")
        good = ex["output"]
        # 构造一个低质量版本（缩短、去除关键信息、加重复）
        if len(good) > 40:
            bad = good[: len(good) // 2] + "... 此商品不错。" if random.random() < 0.5 else good.replace("品牌", "").replace("材质", "")
        else:
            bad = good + " 商品还行。"
        # 质量打分保证 good > bad
        if _quality_score(good) > _quality_score(bad):
            preference.append({"prompt": prompt, "chosen": good, "rejected": bad})
        else:
            preference.append({"prompt": prompt, "chosen": bad, "rejected": good})
    return preference


# ---------------------------------------------------------------------------
# 5. 多模态数据集（图文对）
# ---------------------------------------------------------------------------

def build_multimodal_dataset(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构建多模态数据：图片 + 商品描述文本。"""
    mm = []
    for p in products:
        lang = p["language"]
        is_zh = lang == "zh"
        instr = "请描述这张商品图片。" if is_zh else "Please describe this product image."
        mm.append(
            {
                "image": p["image_url"],
                "instruction": instr,
                "output": p["description"],
                "product_id": p["product_id"],
                "language": lang,
            }
        )
    return mm


# ---------------------------------------------------------------------------
# 6. 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="电商数据构建脚本")
    parser.add_argument("--output_dir", type=str, default="dataset", help="输出目录")
    parser.add_argument("--num_products", type=int, default=2000, help="每种语言的商品数量")
    parser.add_argument("--num_behavior", type=int, default=5000, help="用户行为记录数量")
    parser.add_argument("--lang", type=str, default="zh,en", help="语言，逗号分隔，如 zh,en")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    langs = [l.strip() for l in args.lang.split(",") if l.strip()]

    all_products = []
    for lang in langs:
        print(f"[1/6] 正在生成 {lang} 商品数据（{args.num_products} 条）...")
        products = generate_products(args.num_products, lang)
        all_products.extend(products)
        raw_path = output_dir / f"products_raw_{lang}.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(products, f, ensure_ascii=False, indent=2)

    print("[2/6] 正在清洗商品数据...")
    clean_products_all = clean_products(all_products)
    clean_path = output_dir / "products_clean.json"
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(clean_products_all, f, ensure_ascii=False, indent=2)
    print(f"       清洗后商品数：{len(clean_products_all)}")

    print("[3/6] 正在生成用户行为数据...")
    behaviors = generate_user_behavior(clean_products_all, args.num_behavior)
    with open(output_dir / "user_behavior.json", "w", encoding="utf-8") as f:
        json.dump(behaviors, f, ensure_ascii=False, indent=2)
    print(f"       行为记录数：{len(behaviors)}")

    print("[4/6] 正在构建 SFT 数据集...")
    sft_data = build_sft_dataset(clean_products_all)
    with open(output_dir / "sft_dataset.json", "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    print(f"       SFT 样本数：{len(sft_data)}")

    print("[5/6] 正在构建 Preference 数据集...")
    pref_data = build_preference_dataset(sft_data)
    with open(output_dir / "preference_dataset.json", "w", encoding="utf-8") as f:
        json.dump(pref_data, f, ensure_ascii=False, indent=2)
    print(f"       Preference 样本数：{len(pref_data)}")

    print("[6/6] 正在构建多模态数据集...")
    mm_data = build_multimodal_dataset(clean_products_all)
    with open(output_dir / "multimodal_dataset.json", "w", encoding="utf-8") as f:
        json.dump(mm_data, f, ensure_ascii=False, indent=2)
    print(f"       多模态样本数：{len(mm_data)}")

    print("\n✅ 数据构建完成，所有文件已保存至:", output_dir.resolve())


if __name__ == "__main__":
    main()
