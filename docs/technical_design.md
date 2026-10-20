# Technical design and implementation boundaries

## Data

`ecom_posttrain.data.prepare_data` constructs synthetic products, behavior, multilingual tasks, and preference pairs. `ecom_posttrain.data.process_data` checks fields/prices, tags products, and writes JSONL splits.

Existing artifacts contain 360 products and 1,080 SFT examples. The processor independently shuffles expanded task records. Training contains 93 of 95 validation product IDs and 51 of 52 test product IDs. Split products before deriving tasks/preferences/multimodal examples, assert disjoint identities, and check duplicate text/images.

## Training

SFT/DPO wrappers adapt arguments to `nano_posttrain` configurations. The core implements LoRA and CPU-reference offload; reference labels are aligned with logits. No fixed VRAM saving or speed penalty is established.

DPO loads `model_name` directly and has no SFT adapter argument. A real SFT-to-DPO chain must explicitly restore the intended policy and reference and verify checkpoint identities. The optional `gamma` adds chosen-response cross-entropy; it is not an explicit KL penalty.

SFT saves under `step_latest`, DPO under `latest`. Adapter-only directories are not complete Hugging Face models. Serving/quantization must reconstruct or merge adapters with the matching base.

## Experimental GRPO

The trainer samples response groups, computes rule rewards, and normalizes advantages. Correct the following before interpreting reward trends:

1. The policy token-log-probability call receives an attention mask where response labels are expected.
2. KL is computed under `torch.no_grad()` with detached policy log-probabilities. Adding it changes the displayed loss but does not constrain policy gradients. The standalone loss also detaches its policy KL term.
3. A signed sampled log-probability difference alone does not prove a correctly implemented KL objective.

Acceptance requires response/padding-mask tests, group-normalization checks, nonzero KL-gradient checks, and verified checkpoint restoration. Retain configurations, per-step logs, and held-out generations. GRPO was not rerun for this documentation update.

## Generation and filtering

`generation.generate_content` formats templates; `mode=tuned` does not load a model. It processes at most `batch_size * 5` products and does not implement real language-aware inference. Compare actual models under matched decoding settings and save product/model identities.

Filtering checks length, sensitive terms, and heuristic factuality. Source products are associated by position. Numeric rules can mistake sales counts for prices; missing brand mentions also trigger warnings. Join by ID, extract typed attributes, and measure false positives/missed errors. A 95% warning rate requires analysis.

## Evaluation

`evaluation.evaluate_model` uses references as predictions. Its outputs demonstrate metric computation. Whitespace tokenization is unsuitable for many Chinese sentences; short copied references can still yield zero higher-order n-gram scores.

`evaluation.ab_test` computes heuristic quality/coverage and deterministic simulated CTR. It does not collect clicks, assign traffic, or perform significance tests. Monitoring/performance helpers also contain illustrative values. See [experiment design](ab_test_design.md) for the proposed validation process.

## Multimodal, serving, and quantization

The alignment demo inserts an image placeholder and computes contrastive loss on random tensors; it does not train a vision encoder or establish visual grounding. Real images, features, alignment training, and image-disjoint evaluation remain future work.

Deployment helpers generate vLLM configuration and launch scaffolding. Caching, load balancing, rate limiting, and quantization need integration tests. Optional quantization backends can fall back to dynamic INT8; fallback does not establish export compatibility or preserved quality.

Measure real endpoint requests after warm-up, recording token counts, concurrency, latency distributions, errors, hardware, and peak GPU memory. Compare quantized/unquantized models on the same holdout.

## Orchestration

The pipeline uses obsolete unqualified modules and mixes package-relative artifacts with repository-root defaults. Child failures do not stop execution, and its summary lists attempted stages as completed. Use qualified modules, one artifact root, checked dependencies, and failure propagation before treating it as an acceptance check.
