#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成结果图表用于 README 更新。"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

out_dir = Path(__file__).resolve().parent / "output"
with open(out_dir / "ab_test_result.json", encoding="utf-8") as f:
    ab = json.load(f)
with open(out_dir / "eval_report.json", encoding="utf-8") as f:
    ev = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle("E-commerce LLM Post-Training Pipeline Results", fontsize=14, fontweight='bold')

# 1) A/B 测试对比
ax = axes[0,0]
cats = ["Control (Baseline)", "Experiment (Fine-tuned)"]
scores = [ab["avg_quality_score"]["control"]*100, ab["avg_quality_score"]["experiment"]*100]
colors = ["#4A90D9", "#F5A623"]
bars = ax.bar(cats, scores, color=colors, width=0.5, edgecolor='white')
ax.set_ylabel("Quality Score (%)")
ax.set_title("A/B Test: Content Quality Lift")
ax.set_ylim(0, max(scores)+20)
for b in bars:
    ax.annotate(f"{b.get_height():.1f}%", xy=(b.get_x()+b.get_width()/2, b.get_height()),
                xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
ax.axhline(y=scores[0], color='#4A90D9', linestyle='--', alpha=0.5)

# 2) 业务代理指标提升
ax = axes[0,1]
metrics = ["质量评分", "CTR 代理", "属性覆盖"]
control = [ab["avg_quality_score"]["control"]*100, ab["simulated_ctr"]["control"]*100, ab["avg_attribute_coverage"]["control"]*100]
experiment = [ab["avg_quality_score"]["experiment"]*100, ab["simulated_ctr"]["experiment"]*100, ab["avg_attribute_coverage"]["experiment"]*100]
x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, control, width, label='Control', color='#4A90D9')
ax.bar(x + width/2, experiment, width, label='Experiment', color='#F5A623')
ax.set_ylabel("Score / %")
ax.set_title("Business Proxy Metrics")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.legend(); ax.set_ylim(0, 120)

# 3) 技术指标（ROUGE/L/BLEU）
ax = axes[1,0]
tech = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
vals = [ev["metrics"]["rouge_1"], ev["metrics"]["rouge_2"], ev["metrics"]["rouge_l"], ev["metrics"]["bleu"]]
bars = ax.barh(tech, [v*100 for v in vals], color=["#50C878","#50C878","#50C878","#FFB347"])
ax.set_xlabel("得分 (0-1)")
ax.set_title("Technical Metrics (Generation Quality)")
ax.set_xlim(0, 110)
for b in bars:
    ax.annotate(f"{b.get_width()/100:.2f}", xy=(b.get_width(), b.get_y()+b.get_height()/2),
                xytext=(3,0), textcoords="offset points", ha='left', va='center', fontsize=9)

# 4) Module Progress
ax = axes[1,1]
modules = ["Data", "SFT", "DPO", "Multimodal", "Generation", "Evaluation", "Deployment"]
status = [100, 100, 100, 100, 100, 100, 100]
colors = ["#50C878" if s==100 else "#FFC107" for s in status]
bars = ax.barh(modules, status, color=colors, edgecolor='white')
ax.set_xlabel("Progress %")
ax.set_title("7 大模块执行进度")
ax.set_xlim(0, 110)
for b in bars:
    ax.annotate(f"{int(b.get_width())}%", xy=(b.get_width(), b.get_y()+b.get_height()/2),
                xytext=(3,0), textcoords="offset points", ha='left', va='center', fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(Path(__file__).resolve().parent / "README_charts.png", dpi=150, bbox_inches="tight")
print("Chart saved to ecommerce_project/README_charts.png")
