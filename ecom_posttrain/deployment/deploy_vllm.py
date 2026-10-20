#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/deployment/deploy_vllm.py
============================================
vLLM 服务化部署脚本（模块 7：推理优化与部署）

技术要点：
  - vLLM 高性能推理引擎（混合 batching / paged attention）
  - REST API 服务（FastAPI 风格接口）
  - 缓存层（Redis 代理）
  - 负载均衡与扩展机制（模拟配置说明）

运行：
    # 真实环境（需要 vLLM 安装）：
    python -m vllm.entrypoints.api_server --model Qwen/Qwen2.5-0.5B --port 8000

    # 本脚本用于生成部署配置与测试脚本：
    python deployment/deploy_vllm.py --model_path output/ecommerce_sft/latest --port 8000 --output_dir output/deploy_config
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
from pathlib import Path


def generate_deploy_config(args: argparse.Namespace) -> dict:
    return {
        "service": {
            "name": "ecommerce-llm-inference",
            "framework": "vllm",
            "model_path": args.model_path,
            "port": args.port,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.85,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        },
        "inference_config": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 256,
            "repetition_penalty": 1.05,
            "batch_size": args.batch_size,
        },
        "cache": {
            "enabled": True,
            "store": "redis",
            "ttl_seconds": 600,
            "key_prefix": "ecomm_llm_v1",
        },
        "monitoring": {
            "metrics_endpoint": "/metrics",
            "health_check": "/health",
            "log_format": "json",
        },
        "load_balancing": {
            "strategy": "round_robin",
            "replicas": 2,
            "auto_scale_max": 5,
        },
        "security": {
            "rate_limit_rpm": 120,
            "input_filter_enabled": True,
            "content_filter_path": "generation/content_filter.py",
        },
    }


def generate_api_script(output_dir: Path, config: dict) -> None:
    script_content = (
        "#!/usr/bin/env python3\n"
        '# 使用 vLLM 部署的 API 测试脚本\n'
        "import requests, json\n"
        f"url = 'http://localhost:{config['service']['port']}/v1/chat/completions'\n"
        "payload = {\n"
        '    "model": \"ecommerce-llm\",\n'
        '    "messages": [{"role":"user","content":"描述这款商品图片"}],\n'
        '    "max_tokens": 256,\n'
        '    "temperature": 0.7,\n'
        "}\n"
        "resp = requests.post(url, json=payload, timeout=30)\n"
        "print(resp.json())\n"
    )
    path = output_dir / "test_api.py"
    path.write_text(script_content, encoding="utf-8")
    print(f"  测试脚本已生成：{path}")


def generate_start_script(output_dir: Path, config: dict) -> None:
    model_path = config["service"]["model_path"]
    port = config["service"]["port"]
    script = (
        f"python -m vllm.entrypoints.api_server \\\n"
        f"  --model {model_path} \\\n"
        f"  --port {port} \\\n"
        f"  --max-model-len 2048 \\\n"
        f"  --tensor-parallel-size 1 \\\n"
        f"  --gpu-memory-utilization 0.85 \\\n"
        f"  --batch-size {config['inference_config']['batch_size']}\n"
    )
    path = output_dir / "start_vllm.sh"
    path.write_text(script, encoding="utf-8")
    print(f"  启动脚本已生成：{path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM 部署配置生成")
    parser.add_argument("--model_path", type=str, default="output/ecommerce_sft/latest")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="output/deploy_config")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = generate_deploy_config(args)
    with open(out_dir / "vllm_deploy_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    generate_api_script(out_dir, config)
    generate_start_script(out_dir, config)

    print(f"\n✅ 部署配置已生成 -> {out_dir.resolve()}")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    exit(main())
