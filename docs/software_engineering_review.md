# Software engineering design review

Reviewed on 2026-10-20 with `using-agent-skills` to select the workflow and `code-review-and-quality` to assess correctness, readability, architecture, security, and performance. Evidence consists of source inspection and the [retained verification record](verification/README.md). This update documents findings; it does not implement the proposed runtime changes.

## Design assessment

Separating the reusable `nano_posttrain` core from application data, rewards, evaluation, and serving is the correct direction. Scenario-specific SFT/DPO wrappers reuse the core rather than duplicating its loops. Package modules and JSON/JSONL artifacts provide a simple, inspectable prototype.

The current system is nevertheless a collection of demonstrations rather than a validated continuous post-training service. The central design problem is implicit stage contracts: paths, checkpoint format, source-record identity, execution status, and whether outputs are simulated. A successful script exit cannot yet be interpreted as valid model evaluation or a deployed service. Adding orchestration infrastructure would multiply these ambiguities; first make one linear workflow reliable.

## Findings, ordered by impact

### High: orchestration reports success independently of stage success

[run_pipeline.run](../ecom_posttrain/run_pipeline.py#L36) prints a warning after a nonzero subprocess result and returns normally. The final `completed_steps` list is built from skip flags, not successful results. Child invocations use old unqualified module names, while datasets/outputs are placed under the package directory and other commands default to repository-root paths. This allows stale artifacts or failed stages to produce an apparent successful run.

Use one explicit artifact root and package-qualified modules. Stop on a failed required stage, record the real exit status, and validate each required output before starting its consumer. A simple ordered loop with checked subprocess calls is sufficient; no scheduler framework is needed. Acceptance: deliberately failing one stage produces a nonzero pipeline exit and downstream stages do not run or appear as completed.

### High: checkpoint contracts break the intended training chain

The [DPO wrapper](../ecom_posttrain/model/dpo_train.py) builds `DPOConfig` using the base `model_name`; it does not accept or restore the SFT adapter. The pipeline points GRPO/deployment at `latest`, while SFT writes `step_latest`. Adapter-only directories also cannot be assumed to be full Hugging Face models by [quantize_model](../ecom_posttrain/deployment/quantize_model.py).

Make stages consume the actual checkpoint path returned by training. Specify artifact type, base identifier/revision, adapter configuration, and intended frozen reference. Add one explicit reconstruction/merge step before consumers needing full weights. Acceptance: checksum adapter identity at every transition, test SFT-to-DPO initialization, and load the exported model in a fresh process before benchmarking it.

### High: GRPO has an invalid masking call and no KL gradient

In [GRPOTrainer.grpo_loss](../ecom_posttrain/model/grpo_train.py#L248), the policy `_token_logps` receives `attn` instead of response labels. The KL quantity is computed under `no_grad` with a detached policy term. The [standalone objective](../ecom_posttrain/model/grpo_loss.py#L20) also detaches the policy KL term and duplicates objective logic, creating another place for fixes to diverge.

Correct the response mask, make the intended KL estimator differentiable, and keep one canonical objective used by the trainer and tests. Acceptance: masked prompt/padding tokens contribute nothing, reward ordering changes policy gradients as expected, and changing `kl_beta` changes the gradient when policy/reference differ. No reward trend should be interpreted until these tests pass.

### High: evaluation does not measure inference and product identity leaks

[evaluate_model](../ecom_posttrain/evaluation/evaluate_model.py#L225) copies references into predictions. [generate_content](../ecom_posttrain/generation/generate_content.py) uses hard-coded baseline/tuned templates. [process_data](../ecom_posttrain/data/process_data.py#L349) expands tasks before splitting; 51 of 52 unique test products also occur in training. Benchmark values and CTR are formulas, not measured server behavior or events.

Separate demo mode from actual model inference in artifact metadata and report labels. Split source products before all task expansion. Keep model inference separate from pure metric calculation so saved predictions can be independently audited. Acceptance: zero product-ID intersection across splits, saved base/tuned predictions with model identity, and a metric regression check showing that changing predictions changes scores. Retain a small template demo only if clearly labeled.

### Medium: data association and fallback semantics are weak boundaries

[content_filter.main](../ecom_posttrain/generation/content_filter.py) attaches source records by list index; reordering/subsetting can validate text against another product. Numeric factuality checks can interpret sales as price. The measured tuned-template warning count is 19/20, which is a prompt for error analysis, not proof of guardrail quality. Quantization fallbacks also change the requested backend and need explicit reporting.

Join by product ID, reject missing/duplicate IDs at the boundary, and extract typed facts. Return structured violation types instead of relying on text fragments for severity. Distinguish requested/effective quantization method and stop if the fallback is unacceptable. Acceptance: permuted inputs preserve matching, ambiguous records fail visibly, and a labeled factuality set reports both false positives and missed errors.

### Medium: module/configuration surfaces hide drift

Training wrappers duplicate subsets of core CLI flags, then silently discard fields absent from the core dataclasses. Bootstrap code repeats broad exception handling and `sys.path` adjustments. Dependencies point to the core's mutable `@main`; no lock or tracked CI workflow fixes the verified environment. There are also 21 tracked generated `.pyc` files.

Keep the application-to-core dependency direction, but define a small supported configuration boundary and fail on unknown options. Prefer installed package imports over path mutation; remove obsolete bootstrap paths when package installation is tested. Pin a released core version or tested commit and record exact environment versions. Add one CPU CI job for import, data, artifact-contract, and failure-propagation checks; make GPU smoke tests separate. Remove tracked bytecode in a dedicated housekeeping change.

### Medium: deployment configuration is mistaken for enforcement

[deploy_vllm](../ecom_posttrain/deployment/deploy_vllm.py) writes configuration plus example launch/client scripts. Cache, rate-limit, load-balance, and filter fields are declarations; this code does not wire them into a live request path. Launch/client compatibility and model export need an actual integration check.

Choose one supported local serving path, verify a health check and one generated completion, and connect only the protections that the service truly enforces. Treat model paths, generated shell commands, and external requests as trust boundaries. A documented local prototype does not require production authentication infrastructure, but it must not claim those controls are active.

## Five-axis summary

| Axis | What works | Main gap and next check |
| --- | --- | --- |
| Correctness | Real SFT/DPO wrappers execute short runs | Pipeline success, stage continuity, GRPO objective, and model evaluation need contract tests |
| Readability | Scenario modules have recognizable responsibilities | Remove stale names/path bootstraps and duplicate objective logic before extracting abstractions |
| Architecture | Core/application separation is appropriate | Make artifact identity/type, execution status, and simulated-versus-measured provenance explicit |
| Security | Filter logic is visible and inspectable | Source identity is positional; declared service controls are not enforced by a request handler here |
| Performance | Low-parameter training and CPU reference are available | No valid endpoint benchmark; profile real inference/training before tuning batch size or quantization |

No dependency vulnerability scan, adversarial endpoint test, quantization validation, or load test was performed. Security observations are limited to implementation boundaries inspected here.

## Smallest useful improvement sequence

1. Fix failure propagation, module paths, and artifact roots; add one injected-failure pipeline test.
2. Define checkpoint/input-output contracts, product-disjoint data splits, and actual model inference evaluation.
3. Correct and test GRPO masking/KL using one canonical objective; run a short controlled experiment only afterward.
4. Add CPU CI and an isolated installation check, pin the core dependency, then validate one exported model and one local endpoint.
5. Collect real latency/memory/quality data before adding quantization backends, caching, load balancing, distributed training, or workflow infrastructure.

The useful acceptance bar is concrete: failed stages cannot be reported as successful; each model artifact is loadable and identified; datasets are disjoint; metrics consume actual predictions; and tests enforce these boundaries. The proposed changes preserve a small teaching project while making its claims reproducible.
