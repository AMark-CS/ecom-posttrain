# Experiment design

This is a proposed validation plan. The current `ecom_posttrain.evaluation.ab_test` command compares offline templates with simulated CTR; it does not allocate traffic, collect events, or perform significance testing.

## Offline prerequisite

Create product-disjoint splits before expanding tasks. Current fixtures have 54 SFT test examples and 18 preference test examples; they are synthetic and overlap training products.

Generate base/SFT/DPO outputs on identical held-out prompts with matching decoding settings and logged checkpoint identities. Evaluate the complete selected sample rather than the first 20 products used by the template demo. Report sample counts and language/task breakdowns.

Measure factual attribute coverage, unsupported claims, safety failures, repetition, and language-aware overlap. Blind model identity, randomize presentation order, and ask independent reviewers to rate accuracy, completeness, relevance, fluency, and usefulness. Report agreement and uncertainty.

The earlier proposal suggested 200 examples, three reviewers, mean score 3.8/5, and 65% win rate. These are candidate targets, not achieved results. Predeclare acceptance thresholds and investigate the current 19/20 tuned-template warnings before exposing users.

## Proposed online experiment

After offline/serving checks pass, compare the existing model with the verified tuned model using stable user-level assignment. A 50/50 allocation, seven-day observation window, and later 5% holdback are planning choices; none is implemented here.

Log real impressions, clicks, conversions, model/output identities, errors, filter decisions, and latency with defined event windows. Choose one primary metric and specify whether the minimum detectable effect is relative or absolute. The old 3% relative MDE and 5% proxy-lift target referred to different goals.

Compute sample size using measured baseline rate, desired effect, significance level, power, and user-level clustering. The earlier 5,000 impressions per arm has no retained power calculation and is not a justified requirement. Cover weekday/weekend variation and predeclare stopping rules.

## Analysis and guardrails

Report effect sizes and 95% confidence intervals at the assignment unit. A two-sided 5% significance level is a possible predeclared choice. Account for repeated users, multiple outcomes, and repeated looks. An impression-level z-test alone cannot address these violations. Report language/category/source slices with sample counts.

Candidate guardrails from the original proposal:

| Metric | Proposed boundary relative to control |
| --- | --- |
| Block rate | Increase at most 0.5 percentage points |
| Factual consistency | Decrease at most 1 percentage point |
| p95 latency | Increase at most 10 ms |
| Error rate | At most 1% |

These are policy proposals, not measured guarantees. Decide rollback behavior before launch; current scripts do not enforce these thresholds.

## Rollout and feedback

A rollout could progress from 1% to 10% to 50% after each acceptance gate passes. Keep the prior model and verify rollback. Base expansion on sample size and guardrails rather than the calendar alone.

Save experiment identity/dates, model versions, assignment unit, exclusions, event definitions, counts, effect estimates, intervals, guardrails, and decisions. Simulated CTR and illustrative JSON are not observations.

If coverage rises while factuality falls, add factuality-focused negative examples. Inspect data/tokenization for language gaps and decoding/preference construction for diversity loss. Investigate filter false positives before changing thresholds or whitelists. DPO consumes offline preference pairs; GRPO rule rewards are a separate mechanism.
