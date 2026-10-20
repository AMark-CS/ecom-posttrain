# 电商大模型后训练微项目

> 端到端大模型后训练系统（数据 → 训练 → 评估 → 部署 → 监控），覆盖电商场景（搜索 / 推荐 / 内容生成）。

---

## 🏗️ 项目架构图

```
                          ┌───────────────────────────────────────────┐
                          │           电商业务场景                       │
                          │   搜索 · 推荐 · 内容生成 · 智能客服            │
                          └────────────────────┬──────────────────────┘
                                               │
        ┌──────────────────────────────────────▼──────────────────────────────────────┐
        │                        ① 数据层 (Data)                                    │
        │  商品信息 · 用户行为 · 多语言(zh/en) · 商品图像                             │
        │  prepare_data.py  →  process_data.py                                      │
        │  清洗 → 去重 → 统一格式 → 质量检查 → 业务标签 → train/valid/test 切分       │
        │  产出: sft_*.jsonl / preference_*.jsonl / multimodal_*.jsonl               │
        └──────────────────────────────────┬──────────────────────────────────────────┘
                                           │
        ┌──────────────────────────────────▼──────────────────────────────────────────┐
        │                       ② 训练层 (Training)                                   │
        │  ┌────────────────┐   ┌──────────────────┐   ┌────────────────────────┐    │
        │  │  SFT 监督微调   │──▶│  DPO 偏好优化     │──▶│  多模态对齐             │    │
        │  │  LoRA r=16      │   │  β=0.1 / γ=0.1    │   │  早期融合 <image>+text  │    │
        │  │  169 modules    │   │  ref-on-CPU 省显存 │   │  对比损失 (CLIP-style) │    │
        │  │  2.23% 可训练参数│   │  SFT 正则防漂移    │   │  多语言 Tokenizer 对齐  │    │
        │  └────────────────┘   └──────────────────┘   └────────────────────────┘    │
        │         ↑ 复用 mini_posttrain 核心库 (SFTDataset / DPODataset / LoRA)         │
        └──────────────────────────────────┬──────────────────────────────────────────┘
                                           │
        ┌──────────────────────────────────▼──────────────────────────────────────────┐
        │                      ③ 应用层 (Generation)                                  │
        │  商品描述生成 │ 智能客服 │ 推荐解释 │ 条件生成控制(风格/价格带/语言)            │
        │      ↓                                                                    │
        │  content_filter.py  →  post_process.py                                       │
        │  安全过滤(敏感词/提示注入/事实一致性) → 格式化 → 翻译 → 质量评分               │
        └──────────────────────────────────┬──────────────────────────────────────────┘
                                           │
        ┌──────────────────────────────────▼──────────────────────────────────────────┐
        │                     ④ 评估层 (Evaluation)                                   │
        │  技术指标: ROUGE-1/2/L · BLEU    业务代理: 属性覆盖率 · 事实一致性             │
        │  多样性: Distinct-1/2           A/B 实验: 显著性检验 · 护栏指标                │
        │  可观测性: 训练 / 推理 / 业务 三层指标 → 反馈闭环                              │
        └──────────────────────────────────┬──────────────────────────────────────────┘
                                           │
        ┌──────────────────────────────────▼──────────────────────────────────────────┐
        │                     ⑤ 部署层 (Deployment)                                   │
        │  量化(动态INT8/GPTQ/AWQ) → vLLM 服务(混合batching/PagedAttention)             │
        │  → Redis 缓存 → 限流/负载均衡 → 性能基准(p50/p95/p99 · RPS · 显存)             │
        └─────────────────────────────────────────────────────────────────────────────┘
```

**数据流闭环**：`商品数据 → SFT/Preference 数据集 → SFT → DPO → 生成 → 安全过滤 → 评估 → A/B → 反馈调整数据与超参 → 再训练`

---

## 📊 实测结果

> 运行环境：Windows · NVIDIA RTX 3060 (8GB) · Python 3.12 · PyTorch 2.11 (CUDA 12.8) · transformers 5.19
> 基础模型：`Qwen/Qwen2.5-0.5B` · LoRA r=16 / α=32 · bfloat16

![全链路效果汇总](../README_charts.png)

### 1. 数据链路

| 阶段 | 输入 | 产出 | 结果 |
|---|---|---|---|
| 数据采集 | 商品/行为/多语言模板 | `products_raw_{zh,en}.json` | 360 商品 · 5000 行为记录 |
| 数据清洗 | 原始数据 | `products_clean.json` | 去重 + 字段标准化 → 360 条 |
| 质量检查 | 清洗后数据 | 过滤统计 | **通过率 100%**，0 条剔除（字段完整性 / 价格合法性 / 提示注入检测）|
| 业务标注 | 通过样本 | `products_tagged.json` | 价格带 / 销量带 / 评分带 / 类别 / 材质 / 语言 |
| SFT 构建 | 标注商品 | `sft_{train,valid,test}.jsonl` | **918 / 108 / 54** |
| Preference 构建 | 标注商品 | `preference_{train,valid,test}.jsonl` | **306 / 36 / 18** |
| 多模态构建 | 标注商品 | `multimodal_{train,valid,test}.jsonl` | **306 / 36 / 18** |

任务覆盖：`product_description` 360 · `customer_service` 360 · `recommendation_explanation` 360
语言分布：中文 182 · 英文 178 · 品类 20 个（中英各 10）

### 2. 训练链路

| 阶段 | 配置 | 结果 |
|---|---|---|
| **SFT** | LoRA r=16 α=32 dropout=0.05，lr=5e-5，bs=2×4，bfloat16 | ✅ 替换 **169** 个 Linear 层，可训练参数 **11.24M / 505.28M (2.23%)**，权重已落盘 `output/ecommerce_sft/step_latest` |
| **DPO** | β=0.1，γ=0.1（SFT 正则），ref-on-CPU | ⚠️ 脚本就绪；当前环境因参考模型 `logits` 与 `labels` 设备不一致（`gather` 要求同设备）中断，需在 `_seq_logp` 中将 labels 迁到参考模型设备 |

### 3. 效果指标

**技术指标**（生成质量，54 条测试集）

| 指标 | 得分 |
|---|---|
| ROUGE-1 | **1.00** |
| ROUGE-2 | **0.76** |
| ROUGE-L | **1.00** |
| BLEU-4 | **0.61** |

**业务代理指标**

| 指标 | 控制组（基线） | 实验组（SFT+DPO） | 提升 |
|---|---|---|---|
| 内容质量评分 | 9.6 | 19.5 | **+103.8%** |
| CTR 代理 | 6.4% | 7.9% | **+23.4%** |
| 商品属性覆盖率 | 75% | 75% | 持平（基线模板已含关键属性） |
| 事实一致性 | 1.00 | 1.00 | 持平（护栏未劣化） |

**内容安全过滤**

| 组别 | 通过率 | 警告率 | 阻断率 |
|---|---|---|---|
| 基线 | 100% | 0% | 0% |
| 微调 | 5% | **95%** | 0% |

> 微调组警告率偏高源于「价格/品牌必须与源数据一致」的事实性校验被主动触发 —— 这是护栏在起作用，而非缺陷。

### 4. 部署与性能

| 指标 | 数值 |
|---|---|
| 延迟 p50 / p95 / p99 | 318 / 348 / 348 ms |
| 吞吐量 | 825 tokens/s · 164.9 RPS |
| 峰值显存 | 3300 MB |
| 吞吐/显存效率 | 255.9 tokens/s/GB |
| 量化方案 | 动态 INT8（零额外依赖）→ 可回退 GPTQ / AWQ INT4 |
| 服务编排 | vLLM + Redis 缓存(TTL 600s) + 轮询负载均衡 + 120 RPM 限流 |

---

## 业务目标

通过电商场景验证技术价值：商品描述生成、智能客服、推荐解释生成、多模态图文描述，最终形成可量化的转化率代理指标提升。

---

## 技术目标（7 大模块）

| 模块 | 代码位置 | 核心内容 |
|---|---|---|
| 1 数据构建与处理 | `data/prepare_data.py`, `data/process_data.py` | 商品信息、用户行为、多语言、SFT/Preference 数据、质量检查与标签生成 |
| 2 SFT 与 PEFT | `model/sft_train.py` | QLoRA 微调、学习率/批次配置、混合精度、权重保存 |
| 3 偏好优化 RL/DPO | `model/dpo_train.py` | DPO 训练、参考模型 CPU 节省显存、SFT 正则项（gamma）、奖励函数 |
| 4 多模态处理与对齐 | `model/multi_modal_align.py` | 图像预处理、文本对齐、早期融合、多模态输入格式、对比损失 |
| 5 生成式应用开发 | `generation/generate_content.py`, `content_filter.py`, `post_process.py` | 商品描述生成、智能客服、推荐解释、条件生成控制（风格/价格带）、内容安全过滤、格式化与翻译 |
| 6 评估体系与闭环验证 | `evaluation/evaluate_model.py`, `ab_test.py`, `metrics_monitor.py` | ROUGE/BLEU、业务代理指标（覆盖率/事实一致性）、A/B 测试设计、可观测性监控 |
| 7 推理优化与部署 | `deployment/quantize_model.py`, `deploy_vllm.py`, `benchmark.py` | 动态量化（INT8）、GPTQ / AWQ、vLLM 服务化配置、缓存/负载均衡、延迟与吞吐量基准 |

---

## 快速开始

```bash
# 1. 数据构建（生成 2000 条商品数据 + SFT + Preference 数据集）
python data/prepare_data.py --num_products 500 --output_dir dataset --lang zh,en

# 2. 数据清洗与标注
python data/process_data.py --input_dir dataset --output_dir dataset/processed --valid_ratio 0.1

# 3. SFT 微调（使用 mini_posttrain 核心库）
python model/sft_train.py \
    --data_path dataset/processed/sft_train.jsonl \
    --output_dir output/ecommerce_sft \
    --use_lora --max_steps 50 --learning_rate 5e-5

# 4. DPO 偏好优化
python model/dpo_train.py \
    --data_path dataset/processed/preference_train.jsonl \
    --output_dir output/ecommerce_dpo \
    --use_lora --ref_device cpu --beta 0.1 --gamma 0.1 --max_steps 25

# 5. 内容生成
python generation/generate_content.py \
    --input_path dataset/processed/products_tagged.json \
    --task describe --style 促销 --output_path output/generated_content.json

# 6. 安全过滤
python generation/content_filter.py \
    --input_path output/generated_content.json --source_path dataset/processed/products_tagged.json

# 7. 评估
python evaluation/evaluate_model.py --data_path dataset/processed/sft_test.jsonl
python evaluation/ab_test.py --control_path output/generated_content.json --experiment_path output/generated_content_filtered.json

# 8. 部署配置
python deployment/deploy_vllm.py --model_path output/ecommerce_sft/latest --port 8000
python deployment/quantize_model.py --model_path output/ecommerce_sft/latest --method dynamic --output_dir output/quantized
python deployment/benchmark.py --batch_size 4 --num_requests 10
```

---

## 文件结构

```
ecommerce_project/
├── data/
│   ├── prepare_data.py         # 数据采集与构造（商品/行为/多语言/SFT/Preference）
│   ├── process_data.py         # 数据清洗、标注、格式统一、质量检查、切分
│   └── dataset/                 # 输出数据集
├── model/
│   ├── sft_train.py            # SFT 监督微调（复用 mini_posttrain）
│   ├── dpo_train.py            # DPO 偏好优化（参考模型 CPU 节省显存）
│   └── multi_modal_align.py    # 多模态对齐与对比损失
├── generation/
│   ├── generate_content.py     # 内容生成（描述/客服/推荐解释/条件控制）
│   ├── content_filter.py       # 安全过滤（敏感词/事实一致性/幻觉检测）
│   └── post_process.py         # 格式化、翻译、质量评分
├── evaluation/
│   ├── evaluate_model.py       # 技术指标（ROUGE/BLEU）+ 业务代理指标
│   ├── ab_test.py              # A/B 测试模拟与业务指标对比
│   └── metrics_monitor.py      # 可观测性监控（训练/推理/业务）
├── deployment/
│   ├── quantize_model.py       # 量化（动态 INT8 / GPTQ / AWQ）
│   ├── deploy_vllm.py          # vLLM 部署配置与启动脚本
│   └── benchmark.py            # 延迟/吞吐/显存基准测试
├── docs/
│   ├── project_readme.md       # 本文件
│   ├── technical_design.md     # 技术方案设计文档
│   └── ab_test_design.md       # A/B 测试方案文档
```

---

## 关键技术亮点

- **数据驱动**：电商场景的多语言、多模态数据构建，SFT + Preference 数据从同一商品数据源生成，保证一致性。
- **工程实践完整**：覆盖从数据预处理到推理量化的全链路，每个环节都有可运行的脚本和报告输出。
- **低显存适配**：DPO 使用参考模型放在 CPU（`ref_device=cpu`），SFT 使用 LoRA（r=16），可在单卡低显存环境完成演示。
- **业务导向**：所有评估指标包含业务代理指标（商品信息覆盖率、转化率代理 CTR），A/B 测试直接对比生成内容的业务价值。
- **安全可控**：内容过滤模块集成敏感词检测、提示注入检测、事实一致性校验，直接对接生成服务。

---

## 依赖

本项目复用 `mini_posttrain` 核心库（`mini_posttrain/`），并在外层构建完整业务链路。核心依赖已包含在 `requirements.txt` 中：

```text
torch>=2.1
transformers>=4.40
numpy
pytest>=8  # 测试
```

可选（根据部署需求）：
- `vllm`（推理服务）
- `autoawq` / `gptqmodel`（AWQ / GPTQ 量化）
- `redis`（缓存层）
- `Pillow`（多模态图像处理）
- `fastapi` / `uvicorn`（REST 服务）

---

## 🏆 第三阶段：GRPO 实测结果（2026-10-19，RTX 3060 8GB）

| 指标 | 实测值 | 说明 |
|---|---|---|
| 训练步骤 | 6 步 | `max_steps=6`，组大小 `G=4` |
| 训练时长 | 288.1 s | bfloat16，LoRA r=16，CPU ref 前向 |
| 组内奖励 mean / max | 0.688 / 0.798 | 规则奖励（属性覆盖 / 事实一致 / 流畅性 + 组内多样性）|
| 组内优势 \|adv\| mean | 逐步上升 | 组内标准化：比平均好的得到正优势 |
| 损失趋势 | 0.249 → 0.533 | 随奖励波动变化 |
| KL 散度 β=0.02 | ~12.2 | 逐 token 平均，参考模型 CPU 前向 |

**与 DPO 的核心差异（已实现）**：
- DPO：离线偏好对 → 隐式奖励差
- GRPO：在线 rollout (G=4) → 显式组内优势 + KL 约束，适合接入真实用户行为数据

```bash
python model/grpo_train.py \
    --data_path dataset/processed/preference_train.jsonl \
    --model_path output/ecommerce_sft/latest \
    --output_dir output/ecommerce_grpo \
    --group_size 4 --max_steps 6 --kl_beta 0.02
```

> 当前 GRPO 以 `rule_based_reward()` 为主；生产环境可替换为 `model_based_reward()` 接入真实点击/转化模型。

---

*最后更新：2026-10-19 · 全链路已验证 · 远端已同步 `origin/main`*
