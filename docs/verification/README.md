# Local verification record

Date: 2026-10-20 (Asia/Shanghai). All commands ran from the ecom-posttrain repository root.

## Environment and training

Real SFT/DPO runs used Windows, Python 3.12.14, PyTorch 2.11.0+cu128, Transformers 5.19.0, NVIDIA GeForce RTX 3060, and the cached Qwen2.5-0.5B model. The interpreter was `E:/Code/microduck-qwen/.venv/Scripts/python.exe`, with `PYTHONPATH=E:/Code/nano-posttrain`, `HF_HUB_OFFLINE=1`, and `WANDB_DISABLED=1`.

Use the [README commands](../../README.md#reproduce-the-training-checks) to reproduce the two-step wrapper checks. Retained training log excerpts: [SFT](sft_training.txt) and [DPO](dpo_training.txt). The logs show dataset construction, requested optimizer steps, successful checkpoint writing, and wrapper completion. The wrappers' default logging interval is 10, so these two-step runs do not produce per-step loss logs. Checkpoints were inspected locally; they are excluded from the documentation change.

The default Anaconda Python ran the standard-library-only demo modules below. No dependency installation was needed.

## Reproduce the demonstration

```bash
python -m ecom_posttrain.evaluation.evaluate_model --data_path dataset/processed/sft_test.jsonl --output_path docs/verification/eval_report.json
python -m ecom_posttrain.deployment.benchmark --batch_size 4 --num_requests 5 --output_path docs/verification/benchmark_report.json
python -m ecom_posttrain.generation.generate_content --input_path dataset/processed/products_tagged.json --mode baseline --output_path docs/verification/generated_baseline.json
python -m ecom_posttrain.generation.generate_content --input_path dataset/processed/products_tagged.json --mode tuned --output_path docs/verification/generated_tuned.json
python -m ecom_posttrain.generation.content_filter --input_path docs/verification/generated_baseline.json --source_path dataset/processed/products_tagged.json --output_path docs/verification/baseline_filtered.json
python -m ecom_posttrain.generation.content_filter --input_path docs/verification/generated_tuned.json --source_path dataset/processed/products_tagged.json --output_path docs/verification/tuned_filtered.json
python -m ecom_posttrain.evaluation.ab_test --control_path docs/verification/baseline_filtered.json --experiment_path docs/verification/tuned_filtered.json --output_path docs/verification/ab_test_result.json
```

Retained reports:

- [Evaluation](eval_report.json): copied-reference predictions, not model outputs. Original report group labels are preserved.
- [Template comparison](ab_test_result.json): length-derived quality and formula-derived CTR.
- [Formula benchmark](benchmark_report.json): no actual inference, request timing, or GPU measurement.
- [Baseline filter](baseline_filtered_filter_report.json) and [tuned filter](tuned_filtered_filter_report.json): 20 synthetic template examples per group.
- [Artifact audit](artifact_audit.json): checked split sizes and product-ID intersections.

Intermediate generated/filtered records were omitted to avoid duplicating bilingual fixtures; the commands recreate them. These reports are raw demo outputs, not production measurements. No GRPO run, real online experiment, server launch, quantization/export validation, or held-out generation-quality study was performed.

## README figure and engineering review

Rebuild the figure with `python docs/generate_readme_charts.py` (Matplotlib required; rendered here with 3.7.2). It reads the audit counts, training log receipts, and raw template-filter reports in this directory. Counts are shown directly without smoothing or uncertainty intervals. The template panel is labeled separately from real training checks; simulated CTR and throughput are excluded.

The 1920 x 1520 PNG was visually inspected. Generation, malformed-source rejection, English documentation, relative Markdown links, code fences, and `git diff --check` were checked. No new GPU training was needed to render retained results. See [the software engineering review](../software_engineering_review.md) for design findings and proposed acceptance checks.
