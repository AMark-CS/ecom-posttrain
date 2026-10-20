#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/model/reward.py
===================================
奖励函数设计（GRPO 阶段）

支持两种模式：
  1. rule_based：基于商品属性覆盖、事实一致性、流畅性、多样性的规则打分（默认）
  2. model_based：预留接口，传入一个 ``(response, meta) -> float`` 的打分函数
                  （生产环境可接：用户点击率模型 / 转化率模型 / LLM-as-judge）

奖励设计参考 DeepSeekMath / ROLL：
    R = w1 * 覆盖完整性 + w2 * 事实一致性 + w3 * 语言流畅性 + w4 * 多样性
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable

# 默认权重（可通过 config 覆盖）
DEFAULT_WEIGHTS: dict[str, float] = {
    "coverage": 0.35,
    "factual": 0.30,
    "fluency": 0.20,
    "diversity": 0.15,
}


# ---------------------------------------------------------------------------
# 规则奖励
# ---------------------------------------------------------------------------

def rule_based_reward(
    response: str,
    source_meta: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """电商业务代理指标驱动的规则奖励。返回 [0, 1]。"""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    meta = source_meta or {}
    text = (response or "").strip()
    if not text:
        return 0.0
    low = text.lower()

    # 1) 关键属性覆盖率
    keywords = [
        meta.get("brand", ""),
        meta.get("category", ""),
        meta.get("material", ""),
        meta.get("price_band", ""),
    ]
    keywords = [k for k in keywords if k and k not in ("未知", "unknown", "")]
    coverage = (sum(1 for k in keywords if str(k).lower() in low) / len(keywords)) if keywords else 0.0

    # 2) 事实一致性：价格是否与源数据一致（不一致会重罚）
    price = meta.get("price")
    factual = 0.8
    if isinstance(price, (int, float)) and float(price) > 0:
        if str(price) in text or str(int(price)) in text:
            factual = 1.0
        else:
            # 出现其它"价格型"数字 -> 可能是幻觉
            import re
            nums = re.findall(r"\d+(?:\.\d+)?\s*(?:元|块|¥|\$)?", text)
            factual = 0.4 if nums else 0.7

    # 3) 流畅性：重复标点 / 词重复率
    fluency = 0.85
    if text.count("。。。") >= 1 or text.count("...") >= 2:
        fluency -= 0.35
    tokens = text.split()
    if len(tokens) >= 8:
        uniq_ratio = len(set(tokens)) / len(tokens)
        if uniq_ratio < 0.35:
            fluency -= 0.3
        elif uniq_ratio < 0.55:
            fluency -= 0.12
    if len(text) < 15:
        fluency -= 0.25  # 过短通常信息不足

    # 4) 多样性占位（组内差异由 compute_group_diversity 统一补齐）
    diversity = 0.8

    score = (
        w["coverage"] * coverage
        + w["factual"] * factual
        + w["fluency"] * max(0.0, fluency)
        + w["diversity"] * diversity
    )
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# 组内多样性（与组内其它完成比较）
# ---------------------------------------------------------------------------

def compute_group_diversity(completions: list[str]) -> list[float]:
    """组内多样性：每条完成与其它完成的 1-gram Jaccard 距离。"""
    n = len(completions)
    if n <= 1:
        return [1.0]
    grams = [set(c.split()) for c in completions]
    out = []
    for i in range(n):
        others = [grams[j] for j in range(n) if j != i]
        dists = []
        for o in others:
            union = grams[i] | o
            inter = grams[i] & o
            dists.append(1.0 - (len(inter) / len(union) if union else 0.0))
        out.append(sum(dists) / len(dists))
    return out


def apply_group_diversity(
    completions: list[str],
    base_rewards: list[float],
    w_diversity: float = 0.15,
) -> list[float]:
    """把组内多样性并入奖励。"""
    div = compute_group_diversity(completions)
    return [max(0.0, min(1.0, (1 - w_diversity) * r + w_diversity * d))
            for r, d in zip(base_rewards, div)]


# ---------------------------------------------------------------------------
# 组内相对优势（GRPO 核心）
# ---------------------------------------------------------------------------

def compute_group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """DeepSeekMath 风格的组内标准化优势。

    A_i = (r_i - mean(r)) / (std(r) + eps)

    直观含义：比组内平均好的样本得到正优势（被强化），差的得到负优势（被抑制）。
    这就是 GRPO 相对 PPO 的核心简化 —— 不需要训练独立的 value/critic 模型。
    """
    if not rewards:
        return []
    mean_r = statistics.mean(rewards)
    std_r = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return [(r - mean_r) / (std_r + eps) for r in rewards]


# ---------------------------------------------------------------------------
# 模型奖励（预留接口）
# ---------------------------------------------------------------------------

RewardFn = Callable[[str, dict[str, Any]], float]


def model_based_reward(
    scorer: RewardFn,
) -> RewardFn:
    """把任意打分函数包装成统一的 reward 签名。

    生产环境示例::

        def ctr_scorer(response: str, meta: dict) -> float:
            return ctr_model.predict(response, meta["product_id"])

        reward = model_based_reward(ctr_scorer)
    """
    return scorer
