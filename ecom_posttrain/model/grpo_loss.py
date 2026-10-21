"""Response-only GRPO surrogate and differentiable nonnegative KL estimator."""
import torch


def grpo_loss(policy_tok, policy_mask, ref_tok, ref_mask, advantages, kl_beta,
              device=None, *, old_tok=None, clip_epsilon=.2):
    """Original sequence-normalized GRPO; one on-policy update by default."""
    if policy_tok.shape != ref_tok.shape or policy_mask.shape != policy_tok.shape or not torch.equal(policy_mask, ref_mask):
        raise ValueError("Policy and reference response masks must match")
    lengths = policy_mask.sum(-1)
    if (lengths <= 0).any():
        raise ValueError("Every rollout must contain response tokens")
    if kl_beta < 0 or not 0 < clip_epsilon < 1:
        raise ValueError("Invalid KL weight or clipping bound")
    if advantages.shape != lengths.shape or not torch.isfinite(advantages).all():
        raise ValueError("One finite advantage is required per rollout")
    mask = policy_mask.bool()
    old = policy_tok.detach() if old_tok is None else old_tok.detach()
    ratio = torch.exp(torch.where(mask, policy_tok - old, 0.))
    surrogate = torch.minimum(ratio * advantages[:, None],
                              ratio.clamp(1-clip_epsilon, 1+clip_epsilon) * advantages[:, None])
    delta = torch.where(mask, ref_tok.detach() - policy_tok, 0.)
    token_kl = torch.expm1(delta) - delta
    policy_loss = -((surrogate * policy_mask).sum(-1) / lengths).mean()
    kl = ((token_kl * policy_mask).sum(-1) / lengths).mean()
    loss = policy_loss + kl_beta * kl
    if not torch.isfinite(loss):
        raise ValueError("Non-finite GRPO objective")
    return loss, {"loss": loss.detach().item(), "policy_loss": policy_loss.detach().item(), "kl": kl.detach().item()}
