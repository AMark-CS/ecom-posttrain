# 技术方案设计文档

> 电商大模型后训练系统 —— 端到端技术方案

---

## 1. 总体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                         数据层 (Data)                                 │
│  商品信息 · 用户行为 · 多语言 · 图像                                  │
│  prepare_data.py → process_data.py                                  │
│  产出: sft_{train,valid,test}.jsonl / preference_*.jsonl            │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                       训练层 (Training)                               │
│  ┌──────────┐      ┌──────────────┐      ┌──────────────────┐       │
│  │ SFT/LoRA │ ───▶ │ DPO 偏好优化 │ ───▶ │ 多模态对齐增强     │       │
│  │ mini_post│      │ ref-on-CPU   │      │ 早期融合+对比损失  │       │
│  └──────────┘      └──────────────┘      └──────────────────┘       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                      应用层 (Generation)                             │
│  商品描述生成 │ 智能客服 │ 推荐解释 │ 条件生成控制                     │
│  → content_filter（安全）→ post_process（格式化/翻译/质检）           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                    评估层 (Evaluation)                                │
│  技术指标 (ROUGE/BLEU) │ 业务代理指标 │ A/B 实验 │ 可观测性监控         │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                    部署层 (Deployment)                                │
│  量化 (INT8/INT4) → vLLM 服务 → 缓存/限流/负载均衡 → 性能基准        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. 模块 1：数据构建与处理

### 2.1 数据来源设计

| 数据类型 | 字段 | 生产来源（真实环境） | 本项目模拟 |
|---|---|---|---|
| 商品信息 | title/description/image/price/category/sales/rating | 商品中心 MySQL + 商品图片 CDN | `generate_products()` |
| 用户行为 | user_id/product_id/action/timestamp/session/device | 埋点日志（Kafka）+ 用户画像表 | `generate_user_behavior()` |
| 多语言 | language 字段 | 国际化 CMS + 翻译平台 | zh/en 双语模板 |
| 多模态 | image_url + text | 商品图 + 结构化属性 | 图像 URL + 描述文本 |

### 2.2 清洗流水线

```
原始数据 (products_raw_*.json)
    ↓ 去重 (brand|title|category 指纹)
    ↓ 文本清洗 (去 HTML、合并空白)
    ↓ 字段标准化 (价格 float、销量 int、评分 float)
清洗后 (products_clean.json)
    ↓ 质量检查 (字段完整性 / 长度区间 / 价格合法性 / 提示注入)
    ↓ 业务标签 (价格带 / 销量带 / 评分带 / 类别 / 材质 / 语言)
标注后 (products_tagged.json)
    ↓ 统一 schema (messages + meta)
    ↓ 切分 (train / valid / test)
训练集 (sft_*.jsonl, preference_*.jsonl)
```

**关键设计点**：
- 质量检查使用**规则 + 统计双层**：规则层拦截缺失字段、异常价格、提示注入；统计层输出 `data_report.json` 供人工审核。
- 剔除的样本会记录原因（`rejection_reasons`），保证数据链路可追溯。

### 2.3 SFT 数据格式（统一 schema）

```json
{
  "messages": [
    {"role": "user", "content": "请根据以下商品信息生成一段吸引人的商品描述。\n品牌：小米\n类别：电子产品\n..."},
    {"role": "assistant", "content": "这款电子产品由小米推出..."}
  ],
  "meta": {
    "task_type": "product_description",
    "product_id": "PZH000001",
    "language": "zh",
    "tags": {"price_band": "mid_low", "category": "电子产品", "material": "不锈钢"},
    "source": "ecommerce_mock"
  }
}
```

覆盖三类电商任务：
1. `product_description` —— 商品描述生成（内容生成）
2. `customer_service` —— 价格咨询问答（搜索/客服）
3. `recommendation_explanation` —— 推荐理由生成（推荐）

### 2.4 Preference 数据构造

从同一批商品数据构造**优劣对**：

- `chosen`：高质量描述 —— 包含品牌/材质/价格带/使用场景
- `rejected`：低质量描述 —— 信息缺失 + 冗余表达

```
L = -E[ log σ( β·( logπ(y_w|x) - logπ(y_l|x) - logπ_ref(y_w|x) + logπ_ref(y_l|x) ) ) ]
```

---

## 3. 模块 2：SFT 与参数高效微调

### 3.1 训练配置

| 超参 | 值 | 说明 |
|---|---|---|
| base model | Qwen2.5-0.5B | 支持多语言，0.5B 可在单卡低显存完成 |
| method | LoRA (r=16, α=32, dropout=0.05) | 显存 ≈ 全参的 1/4 |
| lr | 5e-5 | LoRA 常用区间 |
| batch | 2 × grad_accum 4 = 8 | 有效批次 |
| precision | bfloat16 | 混合精度 |
| max_length | 1024 | 覆盖中英文描述 |

### 3.2 损失与数据流

```
messages → chat_template(add_generation_prompt=True)  → prompt_ids
messages → chat_template(add_generation_prompt=False) → full_ids
labels   = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]
loss     = CrossEntropy(logits[:, :-1], labels[:, 1:], ignore_index=-100)
```

Prompt 区域完全屏蔽，梯度只来自响应 token —— 这是避免"模型学到用户指令"的关键。

### 3.3 复用 mini_posttrain 核心库

本项目不重复造轮子：训练逻辑直接复用 `mini_posttrain.trainers.SFTTrainer` / `DPOTrainer`，仅在外层做业务数据适配与报告生成。好处是**教学级代码可读性** + **业务链路完整**同时具备。

---

## 4. 模块 3：偏好优化（DPO）

### 4.1 核心技巧：参考模型放 CPU

```
       ┌──────────────────────────────┐
       │            GPU               │
       │  ┌────────────────────────┐  │      ┌──────────────────────┐
       │  │   policy (LoRA r=16)   │  │      │        CPU           │
       │  │   base + A·B (train)   │──┼──────┼─▶ input_ids.cpu()   │
       │  └────────────────────────┘  │      │  ┌────────────────┐  │
       │                              │      │  │  ref (frozen)  │  │
       │                              │◀─────┼──│  base only     │  │
       │  ◀─── logp tensor [B]        │      │  └────────────────┘  │
       └──────────────────────────────┘      └──────────────────────┘
```

因为 `policy = base + LoRA(A,B)`，参考模型就是**去掉 LoRA 的 base**，不需要第二份 GPU 副本。
代价：CPU 前向约慢 3x，但 DPO 中参考前向占总计算的一半，整体约慢 10%，省 ~50% 显存。

### 4.2 关键超参

- `beta = 0.1`：偏好强度。越小越保守（越接近 SFT），越大越激进。
- `gamma = 0.1`：SFT 正则项权重，在 DPO loss 上叠加 `gamma * CE(policy, chosen)`，防止模型偏离基础分布过远（缓解"长度漂移"与"输出坍缩"）。

### 4.3 奖励函数设计

本项目采用**可解释的规则式奖励**（对应业务代理指标）：

```python
score = 0.4 * min(len(text)/200, 1.0)        # 信息量
      + 0.3 * has_key_facts(text)            # 关键属性覆盖（品牌/材质/价格）
      - 0.2 * has_repetition(text)           # 冗余惩罚
```

生产环境可替换为：点击率代理、转化率代理模型打分、或 LLM-as-judge。

### 4.4 安全约束

- **KL 约束**：DPO 隐式包含 KL 惩罚项（参考模型锚点），gamma 提供额外显式约束。
- **输出过滤**：`generation/content_filter.py` 在推理侧兜底，敏感词 / 提示注入 / 事实不一致 → 阻断或降级。

---

## 5. 模块 4：多模态处理与对齐

### 5.1 融合策略对比

| 策略 | 做法 | 电商适用性 |
|---|---|---|
| 早期融合 | 图像 patch embedding 与文本 embedding 在输入层拼接 | 适合图文描述生成，训练成本低 ✅ |
| 中间融合 | 视觉编码器输出投影后注入 LLM 中间层 | 适合细粒度属性抽取，实现复杂度中等 |
| 晚期融合 | 各模态独立编码后融合 | 适合推荐/排序，延迟敏感 |

本项目选择**早期融合**（`<image>` 占位符 + 文本拼接），与 Qwen2.5-VL 系列接口一致。

### 5.2 对齐损失

```python
# CLIP 风格对比损失
logits = image_feat @ text_feat.T / τ
loss = ½( CE(logits, labels) + CE(logits.T, labels) )
```

配合**信息增益损失**（InfoNCE 的变体，惩罚生成中忽略图像内容的情况）。

### 5.3 多语言处理

- 语言检测 → 路由到对应 prompt 模板
- 中英文分别构造 SFT 样本，`meta.language` 标注
- Tokenizer 对齐：共用 Qwen2.5 BPE（中文效率高），无需多 Tokenizer 方案

---

## 6. 模块 5：生成式应用

### 6.1 三类生成任务

| 任务 | 输入 | 输出 | 业务价值 |
|---|---|---|---|
| 商品描述生成 | 结构化商品属性 | 营销文案 | 提升点击率，降低运营成本 |
| 智能客服 | 用户问题 | 回答 | 降低客服人力成本 |
| 推荐解释 | 商品/用户画像 | 解释文本 | 提升推荐点击与信任度 |

### 6.2 条件生成控制

在指令中注入控制标签，实现**无需额外模型**的样式控制：

```
[风格=促销][价格带=mid_low]
品牌：小米
类别：电子产品
请生成一段商品描述。
```

支持维度：风格（正式/友好/促销/专业/亲切/幽默）、价格带、类别、语言。

### 6.3 后处理链路

```
生成 → 安全过滤 → 格式化 → 翻译 → 质量评分 → 导出
```

---

## 7. 模块 6：评估体系

### 7.1 指标体系

**技术指标**（生成质量）
- ROUGE-1 / ROUGE-2 / ROUGE-L
- BLEU-4

**业务代理指标**
- `attribute_coverage`：生成描述覆盖了源商品关键属性的比例（**与转化率最相关的代理指标**）
- `factual_consistency`：价格 / 品牌与源数据的一致性（防幻觉）

**多样性指标**
- Distinct-1 / Distinct-2

### 7.2 A/B 实验

见 `docs/ab_test_design.md`。

### 7.3 可观测性闭环

```
训练指标 (loss, lr, grad_norm)
    ↓
推理指标 (p50/p95/p99 latency, rps, error_rate, GPU util)
    ↓
业务指标 (conversion_rate_proxy, feedback_score, coverage)
    ↓
反馈 → 调整数据/超参 → 再训练 → 再评估
```

---

## 8. 模块 7：推理优化与部署

### 8.1 量化策略

| 方法 | 精度 | 显存 | 速度 | 依赖 |
|---|---|---|---|---|
| 动态量化 | INT8 | ~50% | 中 | torch 原生 ✅ |
| GPTQ | INT4 | ~30% | 高 | auto-gptq / gptqmodel |
| AWQ | INT4 | ~30% | 高 | autoawq |

`quantize_model.py` 在缺少可选依赖时**自动回退**到动态量化，保证本地环境可跑通。

### 8.2 服务化架构

```
                    ┌──────────────┐
   Client ────────▶ │  Nginx/LB    │ ── round_robin / least_conn
                    └──────┬───────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ vLLM #1  │ │ vLLM #2  │ │ vLLM #3  │   ← 混合 batching / paged attention
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             └─────────────┴────────────┘
                           │
                  ┌────────▼────────┐
                  │  Redis Cache    │  ← TTL 600s，key=hash(prompt+params)
                  └─────────────────┘
```

### 8.3 关键工程点

- **vLLM 参数**：`--max-model-len 2048 --gpu-memory-utilization 0.85 --batch-size 8`
- **缓存键**：`sha256(prompt + temperature + max_tokens)`，避免参数不同却命中同一缓存
- **限流**：120 RPM / 实例，超限返回 429
- **降级**：内容过滤命中"block"级 → 返回兜底文案，不影响主链路可用性

---

## 9. 本地环境执行路径

```bash
# 环境
pip install -r requirements.txt

# 端到端（无需 GPU 即可跑通数据/评估/部署配置；训练需 torch）
python data/prepare_data.py --num_products 200 --output_dir dataset
python data/process_data.py --input_dir dataset --output_dir dataset/processed
python model/sft_train.py --data_path dataset/processed/sft_train.jsonl --output_dir output/sft --max_steps 5
python model/dpo_train.py --data_path dataset/processed/preference_train.jsonl --output_dir output/dpo --max_steps 5
python generation/generate_content.py --input_path dataset/processed/products_tagged.json --task describe
python generation/content_filter.py --input_path output/generated_content.json
python generation/post_process.py --input_path output/generated_content_filtered.json
python evaluation/evaluate_model.py --data_path dataset/processed/sft_test.jsonl
python evaluation/ab_test.py --control_path output/generated_content.json --experiment_path output/generated_content_filtered.json
python evaluation/metrics_monitor.py --mode inference
python deployment/deploy_vllm.py --model_path output/sft/latest
python deployment/quantize_model.py --model_path output/sft/latest --method dynamic --output_dir output/quantized
python deployment/benchmark.py --batch_size 4
```

---

## 10. 面试表达要点

1. **数据是 80%**：我设计了从采集→清洗→标注→格式统一→质量检查→切分的完整链路，并把 SFT 与 Preference 数据从同一商品源生成，避免了两套数据的不一致。
2. **后训练全链路理解**：SFT 解决"会做"，DPO 解决"做得好"，两者共用一套数据格式与训练器。
3. **工程 trick 讲得清**：参考模型放 CPU 省 50% 显存，因为 LoRA 下 policy = base + ΔW；prompt masking 用 chat_template 渲染两次取长度差。
4. **业务闭环**：评估指标不止 BLEU/ROUGE，我加入了商品属性覆盖率和事实一致性 —— 这两个才是与转化率真正相关的代理指标。
5. **安全与可观测**：内容过滤做敏感词 + 提示注入 + 事实一致性三层；监控覆盖训练/推理/业务三层指标。
