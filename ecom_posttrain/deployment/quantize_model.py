#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/deployment/quantize_model.py
=============================================
模型量化脚本（模块 7：推理优化）

支持：
  - 动态量化（dynamic）：PyTorch 原生，逐层 INT8，无需校准数据，即时生效
  - 静态量化（static）：GPTQ 风格（若环境中安装了 auto-gptq / gptqmodel 则调用）
  - AWQ / GPTQ 兜底：未安装时自动回退到动态量化，并给出明确提示

运行：
    # 动态量化（推荐本地环境，零额外依赖）
    python deployment/quantize_model.py --model_path output/ecommerce_sft/latest --method dynamic --output_dir output/quantized

    # AWQ（需要 autoawq）
    python deployment/quantize_model.py --model_path Qwen/Qwen2.5-0.5B --method awq --output_dir output/awq_int4
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
import shutil
from pathlib import Path


def _is_loadable_model(path: str) -> bool:
    """判定模型路径是否可用。

    - 本地目录：必须存在且含 ``config.json``
    - Hub id：形如 ``org/name``，交给 transformers 自行下载
    - 其余（不存在的本地路径）：判定为不可用
    """
    raw = str(path).strip()
    p = Path(raw)
    if p.exists():
        if p.is_dir():
            return (p / "config.json").exists()
        if p.is_file():
            return p.suffix in (".json", ".pt", ".bin", ".safetensors")
        return False
    # 不存在的本地路径特征：盘符、相对前缀、绝对路径
    if re.match(r"^[A-Za-z]:[\\/]", raw) or raw.startswith((".", "/", "\\")):
        return False
    # Hub id: org/name
    return bool(re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$", raw))


def get_model_size(path: str | Path) -> float:
    """返回目录大小（MB）。"""
    total = 0
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1024 / 1024
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / 1024 / 1024


def dynamic_quantize(model_path: str, output_dir: str) -> dict:
    """动态量化：INT8 权重，无校准数据，即时生效。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("🔧 执行动态量化 (Dynamic INT8)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"   已保存到 {out}")
    return {"method": "dynamic_int8", "output_dir": str(out)}


def static_quantize_gptq(model_path: str, output_dir: str, bits: int = 4) -> dict:
    """GPTQ 静态量化（需要 auto-gptq / gptqmodel）。"""
    try:
        from gptqmodel import GPTQModel, QuantizeConfig  # type: ignore
    except ImportError:
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore
            from transformers import AutoTokenizer

            print("🔧 执行 GPTQ 静态量化 (INT4)...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = AutoGPTQForCausalLM.from_quantized(
                model_path, device="cpu", use_safetensors=True
            )
            quantize_config = BaseQuantizeConfig(
                bits=bits, group_size=128, desc_act=False
            )
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            model.quantize(quantize_config)
            model.save_pretrained(out, save_safetensors=True)
            tokenizer.save_pretrained(out)
            return {"method": f"gptq_int{bits}", "output_dir": str(out)}
        except ImportError:
            print("⚠️  未检测到 auto-gptq / gptqmodel，回退到动态量化。")
            return dynamic_quantize(model_path, output_dir)

    from transformers import AutoTokenizer

    print("🔧 执行 GPTQ 静态量化 (INT4, gptqmodel)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    quantize_config = QuantizeConfig(bits=bits, group_size=128, desc_act=False)
    model = GPTQModel.load(model_path, quantize_config)
    model.quantize(tokenizer, quantize_config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out)
    tokenizer.save_pretrained(out)
    return {"method": f"gptq_int{bits}", "output_dir": str(out)}


def awq_quantize(model_path: str, output_dir: str) -> dict:
    """AWQ 量化（需要 autoawq）。"""
    try:
        from awq import AutoAWQForCausalLM  # type: ignore
        from awq.utils import quantize_config  # type: ignore
        from transformers import AutoTokenizer
    except ImportError:
        print("⚠️  未检测到 autoawq，回退到动态量化（INT8）。")
        return dynamic_quantize(model_path, output_dir)

    print("🔧 执行 AWQ 量化 (INT4)...")
    model = AutoAWQForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    qcfg = quantize_config(
        w_bit=4, q_group_size=128, w_sym=True, w_activation_scheme="sym",
        w_group_only=False, w_prune=True, mesh_search=True, cache_dir=None,
    )
    model.quantize(tokenizer, qcfg, quant_sequential=True, max_seq_len=2048)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safetensors=True)
    tokenizer.save_pretrained(out)
    return {"method": "awq_int4", "output_dir": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description="模型量化脚本")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/quantized")
    parser.add_argument("--method", type=str, default="dynamic",
                        choices=["dynamic", "gptq", "awq"])
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    original_size = get_model_size(args.model_path)
    print(f"原始模型大小：{original_size:.1f} MB")

    # 预检：模型路径必须是可加载的 HF 模型目录（或 Hub id），否则给出可执行的提示
    if not _is_loadable_model(args.model_path):
        print(f"⚠️  模型路径不可用：{args.model_path}")
        print("    原因：该目录不是有效的 HuggingFace 模型（缺少 config.json / 权重文件）。")
        print("    跳过量化。请先完成 SFT 训练，或用 --model_path 指定基础模型名，例如：")
        print("      python deployment/quantize_model.py --model_path Qwen/Qwen2.5-0.5B --method dynamic")
        info = {
            "method": "skipped",
            "reason": "model_path_not_loadable",
            "model_path": args.model_path,
            "original_size_mb": round(original_size, 2),
        }
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "quantization_report.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        return 0

    if args.method == "dynamic":
        info = dynamic_quantize(args.model_path, args.output_dir)
    elif args.method == "gptq":
        info = static_quantize_gptq(args.model_path, args.output_dir, args.bits)
    else:
        info = awq_quantize(args.model_path, args.output_dir)

    quantized_size = get_model_size(info["output_dir"])
    compression = original_size / max(quantized_size, 0.01)
    info.update({
        "original_size_mb": round(original_size, 2),
        "quantized_size_mb": round(quantized_size, 2),
        "compression_ratio": round(compression, 2),
    })

    report_path = Path(args.output_dir) / "quantization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n✅ 量化完成")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    exit(main())
