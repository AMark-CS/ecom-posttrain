#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/evaluation/ab_test.py
========================================
A/B 测试脚本（模块 6：A/B 实验与业务验证）

设计：
  - 对照组（Control）：使用原始模型 / 基线描述
  - 实验组（Experiment）：使用微调后的模型（SFT / DPO）
  - 业务指标：点击率代理、转化率代理、描述覆盖率、用户停留时间代理

运行：
    python evaluation/ab_test.py \
        --control_path output/generated_content.json \
        --experiment_path output/generated_content_filtered.json \
        --output_path output/ab_test_result.json
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
from pathlib import Path
from typing import Any

random.seed(42)


def compute_ab_metrics(control_items: list[dict], experiment_items: list[dict]) -> dict[str, Any]:
    """计算 A/B 测试代理指标。"""
    def avg_coverage(items: list[dict]) -> float:
        scores = []
        for item in items:
            meta = item.get("meta", item.get("formatted_meta", {})) or {}
            cov = meta.get("attribute_coverage")
            if cov is not None:
                scores.append(float(cov))
                continue
            text = item.get("final_text", item.get("output", "")) or ""
            checks = [
                meta.get("brand", ""), meta.get("category", ""),
                meta.get("material", ""), meta.get("price_band", ""),
            ]
            checks = [c for c in checks if c and c not in ("未知", "unknown", "")]
            if not checks:
                scores.append(0.0)
                continue
            hit = sum(1 for c in checks if c in text)
            scores.append(hit / len(checks))
        return sum(scores) / max(len(scores), 1)

    def avg_quality_score(items: list[dict]) -> float:
        scores = []
        for item in items:
            qs = item.get("quality_score")
            if isinstance(qs, dict) and "score" in qs:
                scores.append(float(qs["score"]))
            else:
                # 代理评分：描述长度与信息量
                text = item.get("final_text", item.get("output", ""))
                scores.append(min(len(text) / 300, 1.0))
        return sum(scores) / max(len(scores), 1)

    def simulated_ctr(items: list[dict]) -> float:
        # CTR 代理：高质量内容点击率更高
        q = avg_quality_score(items)
        return round(0.05 + q * 0.15, 3)

    return {
        "control_size": len(control_items),
        "experiment_size": len(experiment_items),
        "avg_attribute_coverage": {
            "control": avg_coverage(control_items),
            "experiment": avg_coverage(experiment_items),
        },
        "avg_quality_score": {
            "control": avg_quality_score(control_items),
            "experiment": avg_quality_score(experiment_items),
        },
        "simulated_ctr": {
            "control": simulated_ctr(control_items),
            "experiment": simulated_ctr(experiment_items),
        },
        "lift_pct": {
            "quality_score": round((avg_quality_score(experiment_items) - avg_quality_score(control_items)) / max(avg_quality_score(control_items), 0.01) * 100, 1),
            "ctr": round((simulated_ctr(experiment_items) - simulated_ctr(control_items)) / max(simulated_ctr(control_items), 0.01) * 100, 1),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B 测试")
    parser.add_argument("--control_path", type=str, default="output/generated_content.json")
    parser.add_argument("--experiment_path", type=str, default="output/generated_content_filtered.json")
    parser.add_argument("--output_path", type=str, default="output/ab_test_result.json")
    args = parser.parse_args()

    with open(args.control_path, encoding="utf-8") as f:
        control = json.load(f)
    with open(args.experiment_path, encoding="utf-8") as f:
        experiment = json.load(f)

    result = compute_ab_metrics(control, experiment)
    result["method"] = "offline_simulated_ab_test"
    result["scenario"] = "ecommerce_content_generation"

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("🧪 A/B 测试结果")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"  报告已保存：{out_path.resolve()}")
    return 0


if __name__ == "__main__":
    exit(main())
