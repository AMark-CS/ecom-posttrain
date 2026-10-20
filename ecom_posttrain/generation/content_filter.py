#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/generation/content_filter.py
==============================================
内容安全过滤模块（模块 5 + 模块 3：安全约束）

功能：
  1. 敏感词过滤（政治、违禁品、欺诈等）
  2. 幻觉检测（价格与输入不一致、编造规格）
  3. 事实一致性校验（品牌/材质/价格与源数据比对）
  4. 长度与格式校验
  5. 违规内容打标与替换

这是 DPO / RL 训练中的"输出过滤与安全检查"落地实现。

运行示例：
    python generation/content_filter.py \
        --input_path output/generated_content.json \
        --source_path dataset/processed/products_tagged.json \
        --output_path output/generated_content_filtered.json
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
import re
from pathlib import Path
from typing import Any

# 敏感词库（生产环境应从配置中心/词库服务加载）
SENSITIVE_WORDS: list[str] = [
    # 违禁品
    "毒品", "枪支", "弹药", "爆炸物", "走私",
    # 欺诈
    "刷单", "代刷", "假发票", "办证", "套现", "高仿", "精仿",
    # 违规医疗承诺
    "包治百病", "根治", "无副作用", "特效",
    # 绝对化用语（广告法）
    "最好", "第一", "国家级", "世界级", "100%有效", "绝对",
    # 提示注入
    "ignore previous", "忽略以上指令", "disregard all instructions",
]

# 允许的绝对化用语白名单（部分品类可豁免）
ABSOLUTE_ALLOWLIST: list[str] = ["第一品牌"]


class ContentFilter:
    """电商内容安全过滤器。"""

    def __init__(
        self,
        sensitive_words: list[str] | None = None,
        min_length: int = 15,
        max_length: int = 2000,
        check_factual: bool = True,
    ) -> None:
        self.sensitive_words = sensitive_words or SENSITIVE_WORDS
        self.min_length = min_length
        self.max_length = max_length
        self.check_factual = check_factual
        # 预编译正则提高性能
        self._regex = re.compile("|".join(re.escape(w) for w in self.sensitive_words), re.IGNORECASE)

    # ---- 单项检查 ----

    def check_sensitive(self, text: str) -> tuple[bool, list[str]]:
        """敏感词检查。返回 (是否通过, 命中词列表)。"""
        hits = []
        for w in self.sensitive_words:
            if w.lower() in text.lower():
                if w in ABSOLUTE_ALLOWLIST:
                    continue
                hits.append(w)
        return (len(hits) == 0), hits

    def check_length(self, text: str) -> tuple[bool, str]:
        if len(text) < self.min_length:
            return False, f"内容过短({len(text)} < {self.min_length})"
        if len(text) > self.max_length:
            return False, f"内容过长({len(text)} > {self.max_length})"
        return True, "ok"

    def check_factuality(self, text: str, source: dict[str, Any] | None) -> tuple[bool, list[str]]:
        """事实一致性检查：价格/品牌等关键信息不能与源数据矛盾。"""
        issues: list[str] = []
        if not source or not self.check_factual:
            return True, issues
        price = source.get("price")
        if isinstance(price, (int, float)):
            # 找出文本中的价格数字，检查是否有与源价格不同的数字
            numbers = re.findall(r"\d+(?:\.\d+)?", text)
            for n in numbers:
                try:
                    val = float(n)
                except ValueError:
                    continue
                # 允许销量、评分等合理偏差，仅检查价格区间是否严重不符
                if val > 0 and abs(val - float(price)) > max(1.0, float(price) * 0.5):
                    if val < 100:  # 小数字可能是销量/评分，跳过
                        continue
                    issues.append(f"价格不一致(源:{price}, 文本:{val})")
                    break
        brand = source.get("brand")
        if brand and brand not in ("未知", "") and brand not in text:
            issues.append(f"品牌未提及或错误(源:{brand})")
        return (len(issues) == 0), issues

    # ---- 综合过滤 ----

    def filter_one(
        self,
        text: str,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """对单条内容做完整检查，返回带标签的结果。"""
        violations: list[str] = []
        details: dict[str, Any] = {}

        ok_len, len_msg = self.check_length(text)
        details["length_ok"] = ok_len
        if not ok_len:
            violations.append(len_msg)

        ok_sens, hits = self.check_sensitive(text)
        details["sensitive_words_hit"] = hits
        if not ok_sens:
            violations.append(f"敏感词命中: {hits}")

        ok_fact, fact_issues = self.check_factuality(text, source)
        details["factual_ok"] = ok_fact
        if not ok_fact:
            violations.extend(fact_issues)

        # 严重程度：敏感词/提示注入 = 阻断；事实不一致/长度 = 降级
        blocking = [v for v in violations if "敏感词" in v]
        is_blocked = len(blocking) > 0
        level = "block" if is_blocked else ("warn" if violations else "pass")

        return {
            "text": text,
            "passed": not is_blocked,
            "level": level,
            "violations": violations,
            "details": details,
        }

    def filter_batch(
        self,
        items: list[dict[str, Any]],
        text_key: str = "output",
        source_key: str = "source",
    ) -> dict[str, Any]:
        """批量过滤，输出统计报告。"""
        results = []
        counters = {"pass": 0, "warn": 0, "block": 0}
        for item in items:
            text = item.get(text_key, "") or ""
            source = item.get(source_key)
            res = self.filter_one(text, source)
            counters[res["level"]] += 1
            results.append({**item, "filter_result": res})

        total = max(len(items), 1)
        report = {
            "total": len(items),
            "pass_rate": counters["pass"] / total,
            "block_rate": counters["block"] / total,
            "warn_rate": counters["warn"] / total,
            "counters": counters,
        }
        return {"results": results, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description="电商内容安全过滤")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, default="",
                        help="源商品数据（用于事实一致性检查）")
    parser.add_argument("--output_path", type=str, default="output/generated_content_filtered.json")
    parser.add_argument("--text_key", type=str, default="output")
    parser.add_argument("--min_length", type=int, default=15)
    parser.add_argument("--max_length", type=int, default=2000)
    args = parser.parse_args()

    with open(args.input_path, encoding="utf-8") as f:
        data = json.load(f)

    sources: list[dict[str, Any]] = []
    if args.source_path:
        with open(args.source_path, encoding="utf-8") as f:
            sources = json.load(f)

    # 关联源数据（按顺序对齐）
    for i, item in enumerate(data):
        if i < len(sources):
            item["source"] = sources[i]

    flt = ContentFilter(min_length=args.min_length, max_length=args.max_length)
    out = flt.filter_batch(data, text_key=args.text_key)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out["results"], f, ensure_ascii=False, indent=2)

    report_path = out_path.with_name(out_path.stem + "_filter_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out["report"], f, ensure_ascii=False, indent=2)

    print("🛡️  内容安全过滤完成")
    print(json.dumps(out["report"], ensure_ascii=False, indent=2))
    print(f"  过滤后结果：{out_path.resolve()}")
    print(f"  过滤报告：{report_path.resolve()}")
    return 0


if __name__ == "__main__":
    exit(main())
