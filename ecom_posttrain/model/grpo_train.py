#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_project/model/grpo_train.py
======================================
GRPO 偏好优化（第三阶段：冲上限）

与 DPO 的本质差异
-----------------
DPO  : 离线偏好对 (chosen, rejected) -> 隐式奖励差
GRPO : 在线 rollout，每个 prompt 采样 G 个完成 -> 组内相对优势 -> 显式 KL 约束

GRPO 目标（DeepSeekMath 风格，逐 token 长度归一化形式）：

    L = - 1/G * Σ_i  A_i * (1/|o_i|) * Σ_t log π_θ(o_{i,t} | ·)
        + β * KL(π_ref || π_θ)                          (逐 token 平均)

    A_i = (r_i - mean(r)) / (std(r) + eps)

关键工程点
----------
1. **Rollout 用采样**（temperature > 0, top_p），生成 G 个不同的完成
2. **奖励函数** rule_based_reward：属性覆盖 / 事实一致 / 流畅性 / 组内多样性
3. **优势归一化** 去掉组内均值与方差，只保留相对排序信息
4. **KL 约束** 参考模型放 CPU，policy=base+LoRA，所以 ref 就是 base
5. **长度归一化** 避免长回答天然 logp 更低而被错误惩罚

运行：
    python model/grpo_train.py \
        --data_path dataset/processed/preference_train.jsonl \
        --model_path output/ecommerce_sft/latest \
        --output_dir output/ecommerce_grpo \
        --group_size 4 --max_steps 8 --kl_beta 0.02
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- console / import bootstrap ---
try:
    from ecom_posttrain._console import bootstrap
    bootstrap()
except Exception:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nano_posttrain.models import load_model, load_reference_model, load_tokenizer
from nano_posttrain.utils.logger import get_logger

from ecom_posttrain.model.reward import (
    apply_group_diversity,
    compute_group_advantages,
    rule_based_reward,
)

from ecom_posttrain.model.grpo_config import GRPOConfig, load_prompts

log = get_logger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """Single-GPU GRPO trainer: rollout -> reward -> advantage -> KL-constrained update."""

    def __init__(self, cfg: GRPOConfig) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self._build()

    # ---------------- Build ----------------

    def _build(self) -> None:
        cfg = self.cfg
        log.info("GRPO config: %s", cfg)
        model_id = cfg.model_name
        # 如果检查点目录存在但没有完整 HF 模型（只有 lora.pt），用 base 加载
        if cfg.model_path:
            p = Path(cfg.model_path)
            if p.is_dir() and (p / "lora.pt").exists() and not (p / "config.json").exists():
                log.info("GRPO: loading base %s + LoRA from %s", model_id, cfg.model_path)
                model_id = cfg.model_name
            elif p.is_dir() and (p / "config.json").exists():
                model_id = str(p)
        self.tokenizer = load_tokenizer(model_id)
        dtype = _DTYPE_MAP[cfg.dtype]

        # 策略模型（可训练，LoRA 挂在 base 上）
        self.model = load_model(
            model_id,
            dtype=dtype,
            device=self.device,
            use_lora=cfg.use_lora,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
        )
        # 若Restore from existing LoRA checkpoint
        if cfg.model_path:
            self._maybe_load_lora(cfg.model_path)

        # Reference model (frozen base, on CPU for VRAM savings)
        self.ref_model = load_reference_model(
            cfg.model_name,  # 始终用 base 作为 KL 锚点
            dtype=dtype,
            device=cfg.ref_device,
        )

        if cfg.use_lora:
            from nano_posttrain.optim import freeze_non_lora, mark_only_lora_trainable
            freeze_non_lora(self.model)
            tr, tot = mark_only_lora_trainable(self.model)
            log.info("LoRA trainable params: %d / %d (%.2f%%)", tr, tot, 100 * tr / tot)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
        )

        self.prompts = load_prompts(cfg.data_path, cfg.max_examples)
        log.info("GRPO prompts: %d", len(self.prompts))
        self._wandb = None

    def _maybe_load_lora(self, path: str) -> None:
        """从 SFT/DPO 检查点恢复 LoRA 权重。"""
        lora_file = Path(path) / "lora.pt"
        if not lora_file.exists():
            log.warning("no lora.pt in %s — starting from base + fresh LoRA", path)
            return
        from nano_posttrain.eval.eval_harness import load_lora_into
        load_lora_into(self.model, str(lora_file))

    # ---------------- rollout ----------------

    @torch.no_grad()
    def rollout(self, prompt: str, group_size: int) -> list[str]:
        """Sample G completions for a single prompt."""
        msgs = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        self.model.eval()
        out = self.model.generate(
            enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=group_size,
        )
        completions = []
        for i in range(out.size(0)):
            gen = self.tokenizer.decode(out[i, enc.input_ids.size(1):], skip_special_tokens=True)
            completions.append(gen.strip())
        return completions

    # ---------------- 奖励 ----------------

    def score_group(self, completions: list[str], meta: dict) -> list[float]:
        """Rule-based reward + intra-group diversity."""
        base = [rule_based_reward(c, meta, self.cfg.reward_weights or None) for c in completions]
        return apply_group_diversity(completions, base)

    # ---------------- log-prob ----------------

    def _token_logps(self, model, input_ids: torch.Tensor, labels: torch.Tensor):
        """返回 (逐 token logp [B,T], mask [B,T])。"""
        out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits
        labels = labels.to(logits.device)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        tok = log_probs.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        mask = (shift_labels != -100).float()
        return tok * mask, mask

    def _encode_pair(self, prompt: str, completion: str):
        """Encode (prompt, completion), prompt labels set to -100."""
        tok = self.tokenizer
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        full_text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        if len(prompt_ids) > self.cfg.max_prompt_length:
            return None
        if len(full_ids) > self.cfg.max_length:
            keep = self.cfg.max_length - len(prompt_ids)
            if keep <= 8:
                return None
            full_ids = prompt_ids + full_ids[len(prompt_ids):][:keep]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return full_ids, labels

    def _pad_batch(self, seqs: list[tuple[list[int], list[int]]]):
        max_len = max(len(s[0]) for s in seqs)
        pad_id = self.tokenizer.pad_token_id
        ids, labels, attn = [], [], []
        for i, l in seqs:
            n = len(i)
            ids.append(i + [pad_id] * (max_len - n))
            labels.append(l + [-100] * (max_len - n))
            attn.append([1] * n + [0] * (max_len - n))
        dev = self.device
        return (
            torch.tensor(ids, dtype=torch.long, device=dev),
            torch.tensor(labels, dtype=torch.long, device=dev),
            torch.tensor(attn, dtype=torch.long, device=dev),
        )

    # ---------------- Loss ----------------

    def grpo_loss(
        self,
        seqs: list[tuple[list[int], list[int]]],
        advantages: list[float],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """GRPO Loss：Advantage-weighted length-normalized logp + KL penalty."""
        cfg = self.cfg
        input_ids, labels, attn = self._pad_batch(seqs)

        # 策略前向（保留梯度）
        pol_tok, mask = self._token_logps(self.model, input_ids, attn)
        seq_logp = pol_tok.sum(dim=-1)                       # [B]
        n_tokens = mask.sum(dim=-1).clamp_min(1.0)           # [B]
        norm_logp = seq_logp / n_tokens                      # 长度归一化

        # 参考前向（CPU，无梯度）
        with torch.no_grad():
            ref_ids = input_ids.to(cfg.ref_device)
            ref_labels = labels.to(cfg.ref_device)
            ref_attn = attn.to(cfg.ref_device)
            ref_tok, ref_mask = self._token_logps(self.ref_model, ref_ids, ref_labels)
            ref_tok = ref_tok.to(self.device)
            ref_mask = ref_mask.to(self.device)
            # 逐 token KL 代理：log π_ref - log π_θ
            kl = ((ref_tok - pol_tok.detach()) * mask).sum() / mask.sum().clamp_min(1.0)

        adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        policy_loss = -(adv * norm_logp).mean()
        loss = policy_loss + cfg.kl_beta * kl

        metrics = {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "kl": float(kl.detach()),
            "mean_seq_logp": float(seq_logp.detach().mean()),
            "mean_adv": float(adv.mean()),
            "mean_reward": 0.0,  # 由外部填充
            "max_reward": 0.0,
            "min_reward": 0.0,
        }
        return loss, metrics

    # ---------------- 训练 ----------------

    def fit(self) -> None:
        cfg = self.cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        rng = random.Random(cfg.seed)
        history: list[dict[str, Any]] = []
        t0 = time.time()

        total_steps = cfg.max_steps
        warmup = max(1, int(total_steps * cfg.warmup_ratio))

        for step in range(total_steps):
            # ---- Sample prompts ----
            batch = rng.sample(self.prompts, min(cfg.prompts_per_step, len(self.prompts)))

            all_seqs: list[tuple[list[int], list[int]]] = []
            all_adv: list[float] = []
            all_rewards: list[float] = []

            for item in batch:
                completions = self.rollout(item["prompt"], cfg.group_size)
                rewards = self.score_group(completions, item["meta"])
                advantages = compute_group_advantages(rewards)

                for comp, adv, rew in zip(completions, advantages, rewards):
                    enc = self._encode_pair(item["prompt"], comp)
                    if enc is None:
                        continue
                    all_seqs.append(enc)
                    all_adv.append(adv)
                    all_rewards.append(rew)

            if not all_seqs:
                continue

            # ---- 前向 + Loss ----
            self.model.train()
            self.optimizer.zero_grad()
            loss, metrics = self.grpo_loss(all_seqs, all_adv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], cfg.grad_clip
            )

            # ---- LR scheduling (warmup + cosine)----
            lr = cfg.learning_rate
            if step < warmup:
                lr = cfg.learning_rate * (step + 1) / warmup
            else:
                prog = (step - warmup) / max(1, total_steps - warmup)
                lr = cfg.learning_rate * 0.5 * (1 + float(torch.cos(torch.tensor(prog * 3.1415926))))
            for g in self.optimizer.param_groups:
                g["lr"] = lr
            self.optimizer.step()

            real_rewards = [r for r in all_rewards if r > 0] or all_rewards
            metrics.update({
                "mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
                "max_reward": max(all_rewards) if all_rewards else 0.0,
                "min_reward": min(all_rewards) if all_rewards else 0.0,
                "adv_abs_mean": sum(abs(a) for a in all_adv) / max(len(all_adv), 1),
                "lr": lr,
                "step": step,
                "n_rollouts": len(all_seqs),
            })
            history.append(metrics)
            log.info(
                "step %d/%d | loss=%.4f | policy=%.4f | kl=%.4f | reward=%.3f (max %.3f) | |adv|=%.3f | lr=%.2e",
                step + 1, total_steps, metrics["loss"], metrics["policy_loss"],
                metrics["kl"], metrics["mean_reward"], metrics["max_reward"],
                metrics["adv_abs_mean"], lr,
            )

            if (step + 1) % cfg.save_every == 0 or step + 1 == total_steps:
                self._save()

        self._save()
        # Training history
        with open(Path(cfg.output_dir) / "grpo_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        log.info("GRPO training done in %.1fs", time.time() - t0)

    def _save(self) -> None:
        cfg = self.cfg
        out = Path(cfg.output_dir) / "latest"
        out.mkdir(parents=True, exist_ok=True)
        if cfg.use_lora:
            from nano_posttrain.optim import lora_state_dict
            torch.save(lora_state_dict(self.model), out / "lora.pt")
        else:
            self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        log.info("saved GRPO checkpoint to %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="GRPO 电商场景训练（第三阶段）")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model_path", type=str, default="",
                        help="从该检查点继续训练（SFT/DPO 产物目录）")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/ecommerce_grpo")
    parser.add_argument("--group_size", type=int, default=4, help="每组采样数 G")
    parser.add_argument("--prompts_per_step", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--max_prompt_length", type=int, default=320)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--ref_device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--reward_fn", type=str, default="rule_based",
                        choices=["rule_based", "model_based"],
                        help="奖励来源：rule_based=规则打分；model_based=需接入真实用户反馈模型")
    parser.add_argument("--max_examples", type=int, default=-1)
    args = parser.parse_args()

    cfg = GRPOConfig(
        model_name=args.model_name,
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        group_size=args.group_size,
        prompts_per_step=args.prompts_per_step,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        dtype=args.dtype,
        ref_device=args.ref_device,
        max_examples=args.max_examples,
    )
    trainer = GRPOTrainer(cfg)
    trainer.fit()

    report = {
        "stage": "stage_3_grpo",
        "model_name": args.model_name,
        "model_path": args.model_path,
        "group_size": args.group_size,
        "kl_beta": args.kl_beta,
        "max_steps": args.max_steps,
        "reward_fn": f"{args.reward_fn} (coverage/factual/fluency/diversity)",
        "advantage": "group-relative normalized (DeepSeekMath)",
        "kl": "per-token mean of (log p_ref - log p_policy)",
        "output_dir": args.output_dir,
    }
    with open(Path(args.output_dir) / "grpo_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ GRPO 训练完成 -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
