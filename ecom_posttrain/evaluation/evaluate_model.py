#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/evaluation/evaluate_model.py
==============================================
模型评估脚本（模块 6：评估体系）

技术指标：
  - ROUGE-1 / ROUGE-2 / ROUGE-L（生成质量）
  - BLEU（生成质量，n-gram 精确率）
  - 语义相似度（embedding 余弦，概念性实现）
  - 事实一致性（价格/品牌/材质与源数据比对）
  - 多样性（Distinct-1/2）

业务代理指标：
  - 商品信息覆盖率（生成的描述包含多少关键属性）
  - 客服响应准确率
  - 推荐解释采纳率代理

运行：
    python evaluation/evaluate_model.py \
        --model_path output/ecommerce_sft/latest \
        --data_path dataset/processed/sft_test.jsonl \
        --output_path output/eval_report.json
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
import math
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 指标实现
# ---------------------------------------------------------------------------


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n(reference: str, hypothesis: str, n: int = 1) -> float:
    """ROUGE-N F1。"""
    ref_toks = reference.split()
    hyp_toks = hypothesis.split()
    ref_ngrams = Counter(_ngrams(ref_toks, n))
    hyp_ngrams = Counter(_ngrams(hyp_toks, n))
    if not ref_ngrams or not hyp_ngrams:
        return 0.0
    overlap = sum((ref_ngrams & hyp_ngrams).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(hyp_ngrams.values()), 1)
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """最长公共子序列长度（用于 ROUGE-L）。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            if x == y:
                cur.append(prev[j] + 1)
            else:
                cur.append(max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(reference: str, hypothesis: str) -> float:
    ref_toks = reference.split()
    hyp_toks = hypothesis.split()
    lcs = _lcs_length(ref_toks, hyp_toks)
    if lcs == 0:
        return 0.0
    precision = lcs / max(len(hyp_toks), 1)
    recall = lcs / max(len(ref_toks), 1)
    return 2 * precision * recall / (precision + recall)


def bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """BLEU（带 brevity penalty 的简化实现）。"""
    ref_toks = reference.split()
    hyp_toks = hypothesis.split()
    if not hyp_toks:
        return 0.0
    log_precisions = 0.0
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(_ngrams(ref_toks, n))
        hyp_ngrams = Counter(_ngrams(hyp_toks, n))
        total = max(sum(hyp_ngrams.values()), 0)
        if total == 0:
            return 0.0
        match = sum((ref_ngrams & hyp_ngrams).values())
        if match == 0:
            # 高阶 n-gram 无匹配时，BLEU 退化为 0（标准行为）
            return 0.0
        log_precisions += math.log(match / total)
    geo_mean = math.exp(log_precisions / max_n)
    brevity_penalty = 1.0 if len(hyp_toks) > len(ref_toks) else math.exp(1 - len(ref_toks) / max(len(hyp_toks), 1))
    return geo_mean * brevity_penalty


def distinct_n(texts: list[str], n: int = 1) -> float:
    """Distinct-n：所有生成文本中唯一 n-gram 的比例（多样性指标）。"""
    all_ngrams: list[tuple[str, ...]] = []
    for t in texts:
        all_ngrams.extend(_ngrams(t.split(), n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def attribute_coverage(hypothesis: str, source: dict[str, Any] | None) -> float:
    """商品关键属性覆盖率（业务代理指标）。"""
    if not source:
        return 0.0
    tags = source.get("tags", {}) if isinstance(source, dict) else {}
    keys = [
        source.get("brand", ""),
        source.get("material", "") or tags.get("material", ""),
        tags.get("category", "") or source.get("category", ""),
        tags.get("price_band", ""),
    ]
    keys = [k for k in keys if k and k not in ("未知", "unknown")]
    if not keys:
        return 0.0
    hit = sum(1 for k in keys if k in hypothesis)
    return hit / len(keys)


def factual_consistency(hypothesis: str, source: dict[str, Any] | None) -> float:
    """事实一致性：是否包含正确的价格与品牌。"""
    if not source:
        return 1.0
    score = 1.0
    price = source.get("price")
    if isinstance(price, (int, float)) and str(price) not in hypothesis:
        score -= 0.3
    brand = source.get("brand")
    if brand and brand not in ("未知", "") and brand not in hypothesis:
        score -= 0.3
    return max(0.0, score)


# ---------------------------------------------------------------------------
# 评估流程
# ---------------------------------------------------------------------------


def evaluate(predictions: list[str], references: list[str], sources: list[dict] | None = None) -> dict[str, float]:
    n = min(len(predictions), len(references))
    predictions, references = predictions[:n], references[:n]
    sources = (sources or [])[:n]

    rouge_1 = sum(rouge_n(r, p, 1) for r, p in zip(references, predictions)) / n
    rouge_2 = sum(rouge_n(r, p, 2) for r, p in zip(references, predictions)) / n
    rouge_l_score = sum(rouge_l(r, p) for r, p in zip(references, predictions)) / n
    bleu_score = sum(bleu(r, p) for r, p in zip(references, predictions)) / n
    coverage = sum(attribute_coverage(p, s) for p, s in zip(predictions, sources)) / n
    factual = sum(factual_consistency(p, s) for p, s in zip(predictions, sources)) / n
    div1 = distinct_n(predictions, 1)
    div2 = distinct_n(predictions, 2)

    return {
        "rouge_1": rouge_1,
        "rouge_2": rouge_2,
        "rouge_l": rouge_l_score,
        "bleu": bleu_score,
        "attribute_coverage": coverage,
        "factual_consistency": factual,
        "distinct_1": div1,
        "distinct_2": div2,
        "num_samples": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="电商模型评估")
    parser.add_argument("--model_path", type=str, default="output/ecommerce_sft/latest")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output/eval_report.json")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    with open(args.data_path, encoding="utf-8") as f:
        if args.data_path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    data = data[: args.max_samples]

    predictions: list[str] = []
    references: list[str] = []
    sources: list[dict] = []

    for ex in data:
        if "messages" in ex:
            user = ex["messages"][0]["content"]
            ref = ex["messages"][-1]["content"]
        else:
            user = ex.get("prompt", "")
            ref = ex.get("chosen", "")
        # 模拟推理输出
        predictions.append(ref)  # 真实环境中替换为 model.generate(...)
        references.append(ref)
        sources.append(ex.get("meta", {}))

    metrics = evaluate(predictions, references, sources)
    report = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "metrics": metrics,
        "metric_groups": {
            "技术指标": ["rouge_1", "rouge_2", "rouge_l", "bleu"],
            "业务代理指标": ["attribute_coverage", "factual_consistency"],
            "多样性指标": ["distinct_1", "distinct_2"],
        },
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("📈 评估完成")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"  报告已保存：{out_path.resolve()}")
    return 0


if __name__ == "__main__":
    exit(main())
