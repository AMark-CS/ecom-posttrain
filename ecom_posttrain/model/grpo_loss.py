#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRPO loss (standalone reference)."""
import torch
import torch.nn.functional as F

def grpo_loss(
    policy_tok: torch.Tensor,
    policy_mask: torch.Tensor,
    ref_tok: torch.Tensor,
    ref_mask: torch.Tensor,
    advantages: torch.Tensor,
    kl_beta: float,
    device: str = "cuda",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reference standalone GRPO loss."""
    seq_logp = policy_tok.sum(dim=-1)
    n_tokens = policy_mask.sum(dim=-1).clamp_min(1.0)
    norm_logp = seq_logp / n_tokens
    kl = ((ref_tok - policy_tok.detach()) * policy_mask).sum() / policy_mask.sum().clamp_min(1.0)
    policy_loss = -(advantages * norm_logp).mean()
    loss = policy_loss + kl_beta * kl
    metrics = {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()), "kl": float(kl.detach())}
    return loss, metrics
