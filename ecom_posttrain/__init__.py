"""电商大模型后训练微项目 — 端到端系统。

覆盖模块：
  1. data/process_data.py       数据构建与处理
  2. model/sft_train.py         监督微调与参数高效微调
  3. model/dpo_train.py         偏好优化（DPO）
  4. model/multi_modal_align.py 多模态处理与对齐
  5. generation/*               生成式应用开发
  6. evaluation/*               评估体系与闭环验证
  7. deployment/*               推理优化与部署
"""

__version__ = "0.1.0"
