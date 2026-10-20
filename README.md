# ecom-posttrain

E-commerce scenario LLM post-training demo (SFT → DPO → GRPO → Multi-modal → Generation → Evaluation → Deployment).

## Dependency

Requires the core library from GitHub:

```bash
pip install git+https://github.com/AMark-CS/nano-posttrain.git
```

## Quick Start

```bash
python -m ecom_posttrain.model.sft_train ...
python -m ecom_posttrain.run_pipeline
```

## Structure

- `ecom_posttrain/model/` — SFT / DPO / GRPO / multi-modal trainers
- `ecom_posttrain/data/` — data preparation
- `ecom_posttrain/generation/` — content generation + filtering
- `ecom_posttrain/evaluation/` — A/B test + metrics
- `ecom_posttrain/deployment/` — vLLM deploy / quantize
