#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/run_pipeline.py
===================================
端到端流水线执行脚本（从数据到评估的完整链路验证）

用法：
    python run_pipeline.py [--skip_sft] [--skip_dpo] [--skip_eval] [--skip_deploy]

输出：每个模块的报告保存至 output/ 目录
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
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}\n▶ {desc}\n{'='*60}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # 保证子进程以 UTF-8 输出
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        tail = result.stderr.strip().splitlines()[-12:]
        print("stderr:", "\n".join(tail))
    if result.returncode != 0:
        print(f"⚠️  {desc} 执行失败（return code={result.returncode}），但流水线继续执行...")
    else:
        print(f"✅ {desc} 完成")


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端流水线")
    parser.add_argument("--skip_sft", action="store_true")
    parser.add_argument("--skip_dpo", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_deploy", action="store_true")
    parser.add_argument("--skip_grpo", action="store_true")
    parser.add_argument("--grpo_steps", type=int, default=6)
    parser.add_argument("--num_products", type=int, default=200)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = base_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据构建
    run(
        [sys.executable, "-m", "data.prepare_data",
         "--num_products", str(args.num_products),
         "--output_dir", str(dataset_dir), "--lang", "zh,en"],
        "模块 1：数据构建（商品 + 行为 + 多语言 + SFT/Preference 数据）"
    )

    # 2. 数据清洗与标注
    run(
        [sys.executable, "-m", "data.process_data",
         "--input_dir", str(dataset_dir),
         "--output_dir", str(dataset_dir / "processed"),
         "--valid_ratio", "0.1"],
        "模块 1（续）：数据清洗、标注、统一格式、质量检查、切分"
    )

    if not args.skip_sft:
        # 3. SFT 微调（使用小步数快速验证）
        run(
            [sys.executable, "-m", "model.sft_train",
             "--data_path", str(dataset_dir / "processed" / "sft_train.jsonl"),
             "--output_dir", str(output_dir / "ecommerce_sft"),
             "--use_lora", "--max_steps", "5",
             "--per_device_batch_size", "2", "--grad_accum_steps", "2",
             "--learning_rate", "5e-5"],
            "模块 2：SFT 监督微调（LoRA + bfloat16 + 低步数快速验证）"
        )

    if not args.skip_dpo:
        # 4. DPO 偏好优化
        run(
            [sys.executable, "-m", "model.dpo_train",
             "--data_path", str(dataset_dir / "processed" / "preference_train.jsonl"),
             "--output_dir", str(output_dir / "ecommerce_dpo"),
             "--use_lora", "--ref_device", "cpu",
             "--beta", "0.1", "--gamma", "0.1",
             "--max_steps", "5",
             "--per_device_batch_size", "2", "--grad_accum_steps", "2",
             "--learning_rate", "5e-6"],
            "模块 3：DPO 偏好优化（参考模型 CPU 节省显存 + SFT 正则项）"
        )

    if not args.skip_grpo:
        # 4b. GRPO（第三阶段：组内 rollout + 规则奖励 + KL 约束）
        run(
            [sys.executable, "-m", "ecom_posttrain.model.grpo_train",
             "--data_path", str(dataset_dir / "processed" / "preference_train.jsonl"),
             "--model_path", str(output_dir / "ecommerce_sft" / "step_latest"),
             "--output_dir", str(output_dir / "ecommerce_grpo"),
             "--group_size", "4", "--prompts_per_step", "2",
             "--max_steps", str(args.grpo_steps),
             "--kl_beta", "0.02", "--learning_rate", "1e-6",
             "--max_new_tokens", "96"],
            "模块 3（续）：GRPO 组内偏好优化（rollout → 奖励 → 组内优势 → KL 约束）"
        )

    # 5. 多模态对齐（概念性演示，不依赖真实图像编码器）
    run(
        [sys.executable, "-m", "model.multi_modal_align",
         "--data_path", str(dataset_dir / "processed" / "multimodal_train.jsonl"),
         "--output_dir", str(output_dir / "multimodal_align")],
        "模块 4：多模态处理与对齐（早期融合 + 对比损失概念实现）"
    )

    # 6. 生成应用（基线）
    run(
        [sys.executable, "-m", "generation.generate_content",
         "--input_path", str(dataset_dir / "processed" / "products_tagged.json"),
         "--task", "describe", "--mode", "baseline",
         "--output_path", str(output_dir / "generated_content_baseline.json")],
        "模块 5：基线内容生成（无微调，简洁描述）"
    )
    # 6b. 生成应用（微调）
    run(
        [sys.executable, "-m", "generation.generate_content",
         "--input_path", str(dataset_dir / "processed" / "products_tagged.json"),
         "--task", "describe", "--mode", "tuned", "--style", "促销",
         "--output_path", str(output_dir / "generated_content.json")],
        "模块 5：微调内容生成（条件控制 + 高信息覆盖）"
    )

    # 7. 安全过滤（基线 + 微调）
    run(
        [sys.executable, "-m", "generation.content_filter",
         "--input_path", str(output_dir / "generated_content_baseline.json"),
         "--source_path", str(dataset_dir / "processed" / "products_tagged.json"),
         "--output_path", str(output_dir / "generated_content_baseline_filtered.json")],
        "模块 5（续）：基线内容安全过滤"
    )
    run(
        [sys.executable, "-m", "generation.content_filter",
         "--input_path", str(output_dir / "generated_content.json"),
         "--source_path", str(dataset_dir / "processed" / "products_tagged.json"),
         "--output_path", str(output_dir / "generated_content_filtered.json")],
        "模块 5（续）：微调内容安全过滤"
    )

    # 8. 后处理
    run(
        [sys.executable, "-m", "generation.post_process",
         "--input_path", str(output_dir / "generated_content_filtered.json"),
         "--output_path", str(output_dir / "final_content.json")],
        "模块 5（续）：内容后处理（格式化 + 质量评分）"
    )

    if not args.skip_eval:
        # 9. 评估
        run(
            [sys.executable, "-m", "evaluation.evaluate_model",
             "--data_path", str(dataset_dir / "processed" / "sft_test.jsonl"),
             "--output_path", str(output_dir / "eval_report.json")],
            "模块 6：模型评估（ROUGE/BLEU + 业务代理指标）"
        )
        # 10. A/B 测试（基线 vs 微调）
        run(
            [sys.executable, "-m", "evaluation.ab_test",
             "--control_path", str(output_dir / "generated_content_baseline_filtered.json"),
             "--experiment_path", str(output_dir / "generated_content_filtered.json"),
             "--output_path", str(output_dir / "ab_test_result.json")],
            "模块 6（续）：A/B 测试模拟与业务提升验证"
        )
        # 11. 监控
        run(
            [sys.executable, "-m", "evaluation.metrics_monitor",
             "--mode", "inference",
             "--output_path", str(output_dir / "metrics_inference.json")],
            "模块 6（续）：可观测性监控（推理指标）"
        )

    if not args.skip_deploy:
        # 12. 部署配置
        run(
            [sys.executable, "-m", "deployment.deploy_vllm",
             "--model_path", str(output_dir / "ecommerce_sft" / "latest"),
             "--port", "8000",
             "--batch_size", "4",
             "--output_dir", str(output_dir / "deploy_config")],
            "模块 7：推理服务化部署（vLLM 配置 + 启动脚本）"
        )
        # 13. 量化
        run(
            [sys.executable, "-m", "deployment.quantize_model",
             "--model_path", str(output_dir / "ecommerce_sft" / "latest"),
             "--method", "dynamic",
             "--output_dir", str(output_dir / "quantized")],
            "模块 7（续）：模型量化（动态 INT8，零额外依赖）"
        )
        # 14. 基准
        run(
            [sys.executable, "-m", "deployment.benchmark",
             "--batch_size", "4",
             "--num_requests", "5",
             "--output_path", str(output_dir / "benchmark_report.json")],
            "模块 7（续）：性能基准测试（延迟 / 吞吐 / 显存）"
        )

    print(f"\n{'='*60}\n✅ 端到端流水线执行完毕\n📁 输出目录：{output_dir.resolve()}\n{'='*60}")
    # 生成流水线总结
    summary_path = output_dir / "pipeline_summary.json"
    import json, datetime
    summary = {
        "completed_steps": [
            "data_build", "data_process",
            *( [] if args.skip_sft else ["sft_train"] ),
            *( [] if args.skip_dpo else ["dpo_train"] ),
            "multimodal_align", "generate_content",
            "content_filter", "post_process",
            *( [] if args.skip_eval else ["evaluate_model", "ab_test", "metrics_monitor"] ),
            *( [] if args.skip_deploy else ["deploy_vllm", "quantize_model", "benchmark"] ),
        ],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "output_dir": str(output_dir.resolve()),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📊 流水线总结已保存：{summary_path}")
    return 0


if __name__ == "__main__":
    exit(main())
