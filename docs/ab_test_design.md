# A/B 测试方案设计

> 电商内容生成模型上线验证方案

---

## 1. 实验目标

验证"经过 SFT + DPO 微调的电商内容生成模型"相比基线模型，在**内容质量**与**业务转化代理指标**上是否有显著提升。

**核心假设 (H1)**：微调后模型生成的商品描述在关键属性覆盖率上显著高于基线，且转化率代理指标（CTR proxy）提升 ≥ 5%。

**护栏指标 (Guardrail)**：内容安全过滤拦截率不升高；事实一致性不下降；P95 推理延迟不增加超过 10ms。

---

## 2. 实验设计

### 2.1 分组设计

| 组别 | 模型 | 流量占比 | 说明 |
|---|---|---|---|
| Control | 基座模型（Qwen2.5-0.5B，zero-shot + prompt 工程） | 50% | 现状 |
| Experiment | SFT 模型（+ DPO 偏好优化） | 50% | 本项目产出 |

- **流量切分**：按 `user_id` 哈希分流，保证同一用户始终命中同一组（避免用户感知不一致带来的噪声）。
- **样本量**：每组 ≥ 5000 次曝光（依据 CTR 代理的最小可检测提升 MDE = 3% 反推）。
- **实验周期**：7 天，覆盖工作日与周末的周期性。

### 2.2 流量分配代码

```python
import hashlib

def assign_group(user_id: str) -> str:
    """一致性哈希分流：同一用户固定分组。"""
    h = hashlib.md5(user_id.encode()).hexdigest()
    bucket = int(h[:8], 16) % 100
    return "experiment" if bucket < 50 else "control"
```

### 2.3 反转实验 (Holdback)

上线后保留 5% 流量永久不切到新模型，作为长期对照组，用于检测节假日/大盘趋势对指标的干扰。

---

## 3. 指标体系

### 3.1 主指标 (Primary Metric)

| 指标 | 定义 | 期望方向 |
|---|---|---|
| `ctr_proxy` | 描述曝光后 24h 内点击率（离线用质量分代理） | ↑ |
| `conversion_rate_proxy` | 点击后加购率代理 | ↑ |

### 3.2 护栏指标 (Guardrail Metrics)

| 指标 | 阈值 | 违反处理 |
|---|---|---|
| `block_rate` 内容拦截率 | ≤ Control + 0.5% | 触发回滚 |
| `factual_consistency` | ≥ Control - 1% | 触发回滚 |
| `p95_latency_ms` | ≤ Control + 10ms | 扩容或回滚 |
| `error_rate` | ≤ 1% | 触发回滚 |

### 3.3 观测指标 (Observability)

- `attribute_coverage`：生成描述覆盖的源商品属性比例
- `rouge_l` / `bleu`：与人工标注优质文案的对齐度
- `distinct_2`：多样性，防止输出坍缩
- `user_feedback_score`：用户显式反馈评分

---

## 4. 统计方法

### 4.1 显著性检验

双比例 z 检验（用于 CTR 类比例指标）：

```python
import math

def z_test(p1: float, n1: int, p2: float, n2: int, alpha: float = 0.05):
    """返回 (z 值, p 值, 是否显著)。"""
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0, False
    z = (p1 - p2) / se
    # 双尾 p 值近似
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p, p < alpha
```

- α = 0.05（双尾）
- 最小可检测效应 MDE = 3%（相对提升）
- 同时报告置信区间 95%

### 4.2 多重比较校正

多指标同时观察时使用 Bonferroni 校正：`α' = α / m`（m = 比较的指标数）。

### 4.3 分层分析

- 按品类（10 类）分层，看提升是否来自头部品类
- 按语言（zh / en）分层，验证多语言能力
- 按流量来源分层，排查选择偏差

---

## 5. 离线评估先行（Offline-First）

在进入在线 A/B 之前，先做离线验证（`evaluation/evaluate_model.py` + `ab_test.py`）：

| 阶段 | 数据 | 判定标准 |
|---|---|---|
| 离线 holdout | `sft_test.jsonl`（500 条） | ROUGE-L ≥ 基线 + 5% |
| 离线对比 | 1000 条商品 | attribute_coverage ≥ 基线 + 10% |
| 人工评测 | 200 条盲评 | 专家评分 ≥ 3.8/5，胜率 ≥ 65% |

**离线不通过 → 不进入在线实验**。这避免了消耗真实流量。

---

## 6. 人工评估设计

### 6.1 评审维度（1-5 分）

| 维度 | 说明 |
|---|---|
| 准确性 | 描述与商品实际属性是否一致（无幻觉） |
| 完整性 | 品牌/材质/价格带/场景等关键信息是否齐全 |
| 相关性 | 是否贴合商品品类与目标人群 |
| 吸引力 | 是否有促销力与感染力 |
| 流畅性 | 语言是否通顺、无冗余重复 |

### 6.2 盲评流程

1. 隐去模型来源，随机打乱 Control / Experiment 顺序
2. 3 名评审独立打分
3. 采用**多数投票 + 平均分**裁决
4. 计算 Kendall's W 衡量评审间一致性（< 0.4 说明标准不清晰，需重新对齐标注指南）

---

## 7. 回滚与灰度计划

```
Day 1  灰度 1% 流量 → 观察错误率 / 延迟 / 拦截率
Day 2  灰度 10% → 观察业务指标趋势
Day 3  灰度 50% → A/B 满量运行
Day 4-7 满量 A/B → 显著性检验
Day 8  结论评审 → 全量 / 回滚 / 迭代
```

**自动回滚触发条件**（任一命中立即回滚到 Control）：
- `error_rate > 1%`
- `block_rate` 突增 3 倍
- `p95_latency` 上涨 > 20%

---

## 8. 结果记录模板

```json
{
  "experiment_id": "exp_ecommerce_sft_dpo_v1",
  "start_date": "2024-XX-XX",
  "end_date": "2024-XX-XX",
  "control": {
    "n": 5000,
    "ctr_proxy": 0.112,
    "attribute_coverage": 0.54,
    "factual_consistency": 0.91,
    "block_rate": 0.021
  },
  "experiment": {
    "n": 5000,
    "ctr_proxy": 0.129,
    "attribute_coverage": 0.71,
    "factual_consistency": 0.93,
    "block_rate": 0.019
  },
  "statistics": {
    "metric": "ctr_proxy",
    "z": 3.84,
    "p_value": 0.0001,
    "significant": true,
    "relative_lift_pct": 15.2
  },
  "decision": "ship"
}
```

---

## 9. 闭环反馈

实验结论直接反哺下一轮迭代：

| 观察结果 | 反馈动作 |
|---|---|
| 覆盖率提升但事实性下降 | 增加事实一致性负样本，重新构造 Preference 数据 |
| 英文表现显著弱于中文 | 补齐英文语料，Language-aware 数据配比 |
| 多样性下降 | 在 DPO reward 中加入多样性奖励项 |
| 拦截率偏高 | 扩充白名单 / 优化敏感词库 |

对应代码位置：`data/prepare_data.py`（数据配比）、`model/dpo_train.py`（reward 调整）、`generation/content_filter.py`（词库）。
