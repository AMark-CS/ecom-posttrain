# E-commerce post-training project notes

The [main README](../README.md) is the authoritative summary of current capabilities, verified results, and optimization priorities. The application depends on the separate `nano_posttrain` training core.

## Scope and workflow

The demo covers product descriptions, customer-service responses, and recommendation explanations using synthetic Chinese/English product data. Documentation is in English; bilingual task data is retained.

The intended workflow is:

```text
products -> cleaning/tagging -> product-level split -> task expansion
         -> SFT -> preference optimization -> model inference
         -> filtering -> held-out evaluation -> serving validation
```

Current implementation gaps: tasks are expanded before splitting; DPO starts from the base model; generation uses templates; multimodal alignment uses random features; serving metrics use arithmetic formulas. A generated deployment configuration or pipeline summary does not prove a service ran successfully.

## Evidence

The 2026-10-20 verification completed two-step SFT and DPO checks on an RTX 3060 using the application wrappers and sibling core package. CPU-reference DPO completed. These checks establish execution, not model quality.

[Verification reports](verification/README.md) reproduce the offline demonstration and label simulated inputs. Historical claims without retained logs, including the six-step GRPO result, are excluded from measured results. The old chart is not used as evidence.

## Next work

Use the [optimization priorities](../README.md#optimization-priorities), [technical design](technical_design.md), and [experiment design](ab_test_design.md) to define acceptance checks. This update changes documentation and records evidence; it does not implement the proposed fixes.
