"""Single-GPU, one-update-per-rollout GRPO with exact sampled-token replay."""
from __future__ import annotations
import argparse
from dataclasses import fields
import json
import math
from pathlib import Path
import random
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import StoppingCriteria, StoppingCriteriaList
from nano_posttrain.models import load_model, load_reference_model, load_tokenizer
from nano_posttrain.optim import apply_lora, freeze_non_lora, lora_state_dict
from ecom_posttrain.model.grpo_config import GRPOConfig, load_prompts
from ecom_posttrain.model.grpo_loss import grpo_loss
from ecom_posttrain.model.reward import rule_based_reward, apply_group_diversity, compute_group_advantages


def restore_lora(model, path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    expected = {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}
    renamed = {f"{k}.weight": v for k, v in saved.items()}
    if not expected or renamed.keys() != expected.keys() or any(renamed[k].shape != v.shape for k,v in expected.items()):
        raise ValueError("Adapter tensors do not match the policy/reference model")
    model.load_state_dict(renamed, strict=False)


class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig, *, on_step=None, should_stop=None, on_phase=None):
        self.cfg = self.config = cfg
        self.on_step, self.should_stop = on_step, should_stop or (lambda: False)
        self.on_phase = on_phase or (lambda message: None)
        self.completed_steps = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self._build()

    def _build(self):
        cfg = self.cfg
        model_id = cfg.model_name
        if cfg.model_path and (Path(cfg.model_path)/"config.json").exists():
            model_id = cfg.model_path
        dtype = getattr(torch, cfg.dtype) if self.device == "cuda" else torch.float32
        self.tokenizer = load_tokenizer(model_id)
        self.model = load_model(model_id, dtype=dtype, device=self.device, use_lora=cfg.use_lora,
                                lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
        self.ref_model = load_reference_model(model_id, dtype=dtype, device=cfg.ref_device)
        if cfg.model_path and not (Path(cfg.model_path)/"config.json").exists():
            if not cfg.use_lora:
                raise ValueError("An adapter checkpoint requires use_lora")
            self.ref_model = apply_lora(self.ref_model, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=0)
            for model in (self.model, self.ref_model):
                restore_lora(model, Path(cfg.model_path)/"lora.pt")
            self.ref_model.to(cfg.ref_device)
        self.ref_model.requires_grad_(False).eval()
        if cfg.use_lora:
            freeze_non_lora(self.model)
        self.model.eval()  # Disable dropout for consistent rollout/policy probabilities, keeping gradients enabled.
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(.9,.95))
        self.prompts = load_prompts(cfg.data_path, cfg.max_examples)
        for item in self.prompts:
            text = self.tokenizer.apply_chat_template([{"role":"user","content":item["prompt"]}], tokenize=False, add_generation_prompt=True)
            item["input_ids"] = self.tokenizer(text, add_special_tokens=False).input_ids
        self.prompts = [item for item in self.prompts if 0 < len(item["input_ids"]) <= cfg.max_prompt_length]
        if not self.prompts:
            raise ValueError("No prompts fit the configured prompt budget")
        self.dataset = self.prompts

    @torch.no_grad()
    def rollout(self, prompt_ids, group_size):
        trainer = self
        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return trainer.should_stop()
        inputs = torch.tensor([prompt_ids], device=self.device)
        self.model.eval()
        outputs = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=self.cfg.max_new_tokens, do_sample=True, temperature=self.cfg.temperature,
            top_p=self.cfg.top_p, num_return_sequences=group_size, pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([Stop()]), use_cache=True)
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, (list,tuple)) else [eos]
        sequences, texts = [], []
        for output in outputs:
            completion = output[len(prompt_ids):].tolist()
            for index, token in enumerate(completion):
                if token in eos:
                    completion = completion[:index+1]
                    break
            if not completion:
                raise ValueError("Generation produced no response tokens")
            sequences.append((prompt_ids + completion, [-100]*len(prompt_ids) + completion))
            texts.append(self.tokenizer.decode(completion, skip_special_tokens=True).strip())
        return sequences, texts

    def score_group(self, completions, meta):
        rewards = apply_group_diversity(completions, [rule_based_reward(c,meta,self.cfg.reward_weights or None) for c in completions])
        if not all(math.isfinite(r) for r in rewards):
            raise ValueError("Reward values must be finite")
        return rewards

    def _pad_batch(self, seqs):
        size = max(len(ids) for ids,_ in seqs)
        ids, labels, attention = [], [], []
        for sequence, target in seqs:
            pad = size-len(sequence)
            ids.append(sequence + [self.tokenizer.pad_token_id]*pad)
            labels.append(target + [-100]*pad)
            attention.append([1]*len(sequence)+[0]*pad)
        return tuple(torch.tensor(value, device=self.device) for value in (ids,labels,attention))

    def _token_logps(self, model, input_ids, labels, attention_mask=None):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1].float() / self.cfg.temperature
        target = labels[:, 1:].to(logits.device)
        logps = F.log_softmax(logits, -1).gather(-1,target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return logps, (target != -100).float()

    def grpo_loss(self, seqs, advantages):
        ids, labels, attention = self._pad_batch(seqs)
        policy, mask = self._token_logps(self.model,ids,labels,attention)
        with torch.no_grad():
            reference, ref_mask = self._token_logps(self.ref_model,ids.to(self.cfg.ref_device),labels.to(self.cfg.ref_device),attention.to(self.cfg.ref_device))
        return grpo_loss(policy,mask,reference.to(self.device),ref_mask.to(self.device),
                         torch.tensor(advantages,device=self.device),self.cfg.kl_beta,clip_epsilon=self.cfg.clip_epsilon)

    def fit(self):
        cfg = self.cfg
        rng, history = random.Random(cfg.seed), []
        start = time.perf_counter()
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        for index in range(cfg.max_steps):
            if self.should_stop():
                break
            sequences, advantages, rewards = [], [], []
            flat_groups = 0
            for prompt_index, item in enumerate(rng.sample(self.prompts, min(cfg.prompts_per_step,len(self.prompts)))):
                self.on_phase(f"GRPO step {index+1}/{cfg.max_steps}: sampling prompt {prompt_index+1}, {cfg.group_size} completions")
                group, texts = self.rollout(item["input_ids"],cfg.group_size)
                if self.should_stop():
                    break
                scores = self.score_group(texts,item["meta"])
                adv = compute_group_advantages(scores)
                flat_groups += int(max(scores)-min(scores) <= 1e-6)
                sequences.extend(group); advantages.extend(adv); rewards.extend(scores)
            if self.should_stop():
                break
            self.optimizer.zero_grad()
            metrics = {"loss":0., "policy_loss":0., "kl":0.}
            # Sequence micro-batches retain the original equal-sequence GRPO normalization on an 8GB GPU.
            for rollout_index, (seq, adv) in enumerate(zip(sequences,advantages)):
                if self.should_stop():
                    break
                self.on_phase(f"GRPO step {index+1}/{cfg.max_steps}: scoring/backpropagating rollout {rollout_index+1}/{len(sequences)}")
                loss, values = self.grpo_loss([seq],[adv])
                (loss / len(sequences)).backward()
                for key in metrics:
                    metrics[key] += values[key]/len(sequences)
            if self.should_stop():
                self.optimizer.zero_grad()
                break
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],cfg.grad_clip, error_if_nonfinite=True)
            warmup = max(1,int(cfg.max_steps*cfg.warmup_ratio))
            lr = cfg.learning_rate * min(1.,(index+1)/warmup) * (.5*(1+math.cos(math.pi*max(0,index-warmup)/max(1,cfg.max_steps-warmup))))
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.step()
            self.completed_steps += 1
            metrics.update(step=self.completed_steps,total_steps=cfg.max_steps,learning_rate=lr,
                mean_reward=sum(rewards)/len(rewards),max_reward=max(rewards),min_reward=min(rewards),
                adv_abs_mean=sum(abs(a) for a in advantages)/len(advantages),flat_groups=flat_groups,n_rollouts=len(sequences))
            history.append(metrics)
            if self.on_step:
                self.on_step(metrics)
            if self.completed_steps % cfg.save_every == 0:
                self._save()
        if self.completed_steps:
            self._save()
        (Path(cfg.output_dir)/"grpo_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        self.elapsed_seconds = time.perf_counter()-start

    def _save(self):
        out = Path(self.cfg.output_dir)/"latest"
        out.mkdir(parents=True,exist_ok=True)
        if self.cfg.use_lora:
            torch.save(lora_state_dict(self.model),out/"lora.pt")
        else:
            self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)


def main():
    parser = argparse.ArgumentParser(description="Online GRPO with rule-based e-commerce rewards")
    for field in fields(GRPOConfig):
        if field.name in {"reward_weights","use_lora","use_wandb","wandb_project"}:
            continue
        parser.add_argument("--"+field.name, type=type(field.default), default=field.default, required=field.name == "data_path")
    parser.add_argument("--use_lora", action="store_true",default=True)
    parser.add_argument("--no_lora",dest="use_lora",action="store_false")
    parser.add_argument("--reward_fn",choices=["rule_based"],default="rule_based")
    args = vars(parser.parse_args()); args.pop("reward_fn")
    cfg = GRPOConfig(**args)
    trainer = GRPOTrainer(cfg)
    trainer.fit()
    report = {"completed_steps":trainer.completed_steps,"requested_steps":cfg.max_steps,
              "elapsed_seconds":trainer.elapsed_seconds,"objective":"one-update sequence-normalized GRPO",
              "reference":"initial policy","reward":"rule-based heuristic, not measured user preference"}
    (Path(cfg.output_dir)/"grpo_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
