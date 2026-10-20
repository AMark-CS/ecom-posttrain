"""Rebuild README_charts.png from retained artifact/log reports; needs Matplotlib."""
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/verification"


def read_json(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def main():
    audit = read_json("artifact_audit.json")
    splits = audit["split_counts"]["sft"]
    counts = [splits[name] for name in ["train", "valid", "test"]]
    assert sum(counts) == 1080 and all(value > 0 for value in counts)
    steps = []
    for stage in ["sft", "dpo"]:
        text = (EVIDENCE / f"{stage}_training.txt").read_text(encoding="utf-8")
        match = re.search(r"training: (\d+) optimizer steps", text)
        assert match is not None and "checkpoint to" in text
        steps.append(int(match.group(1)))
    assert steps == [2, 2]
    filters = [read_json(f"{name}_filtered_filter_report.json") for name in ["baseline", "tuned"]]
    assert all(sum(item["counters"].values()) == item["total"] == 20 for item in filters)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    fig.subplots_adjust(left=.12, right=.96, bottom=.24, top=.84, hspace=.70, wspace=.40)
    fig.suptitle("ecom-posttrain | Artifact and execution audit", x=.09, ha="left",
                 fontsize=21, fontweight="bold", y=.97)
    fig.text(.09, .91, "2026-10-20 · Training: Qwen2.5-0.5B / RTX 3060 · Synthetic product fixtures", fontsize=11)
    ax = axes[0, 0]
    bars = ax.barh([0, 1, 2], counts, color="#0072B2")
    ax.bar_label(bars, padding=5)
    ax.set(title="A  Existing SFT split sizes", xlabel="Task-example count",
           yticks=[0, 1, 2], yticklabels=["Train", "Validation", "Test"], xlim=(0, 1080))
    ax.invert_yaxis()
    ax = axes[0, 1]
    overlap = [audit["sft_product_overlap_with_train"][s] for s in ["valid", "test"]]
    totals = [audit["unique_sft_products"][s] for s in ["valid", "test"]]
    assert all(0 <= a <= b for a, b in zip(overlap, totals))
    ax.barh([0, 1], overlap, color="#9B4F00", hatch="//", label="Also in train")
    ax.barh([0, 1], [b-a for a, b in zip(overlap, totals)], left=overlap,
            color="#D8DDE3", label="Not in train")
    for i, (a, b) in enumerate(zip(overlap, totals)):
        ax.text(a/2, i, f"{a}/{b} overlap", color="white", ha="center", va="center", fontweight="bold")
    ax.set(title="B  Product identity leakage", xlabel="Unique product count",
           yticks=[0, 1], yticklabels=["Validation", "Test"], xlim=(0, 105))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2, frameon=False, fontsize=9)
    ax = axes[1, 0]
    bars = ax.bar([0, 1], steps, color="#0072B2")
    ax.bar_label(bars, labels=["2 + saved adapter", "2 + saved adapter"], padding=6, fontsize=10)
    ax.set(title="C  Real training execution checks", ylabel="Optimizer steps",
           xticks=[0, 1], xticklabels=["SFT", "DPO / CPU ref"], ylim=(0, 3), yticks=[0, 1, 2, 3])
    ax = axes[1, 1]
    left = [0, 0]
    for label, color, hatch in [("pass", "#0072B2", ""),
                                 ("warn", "#9B4F00", "//"),
                                 ("block", "#5D6268", "xx")]:
        values = [item["counters"][label] for item in filters]
        ax.barh([0, 1], values, left=left, color=color, hatch=hatch, label=label.title())
        for i, value in enumerate(values):
            if value:
                ax.text(left[i]+value/2, i, str(value), ha="center", va="center",
                        color="white", fontweight="bold")
        left = [a+b for a, b in zip(left, values)]
    ax.set(title="D  Template demo filter outcomes", xlabel="Synthetic template-output count",
           yticks=[0, 1], yticklabels=["Baseline", "Tuned template"], xlim=(0, 20))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=3, frameon=False, fontsize=9)
    fig.text(.09, .035, "Panels A/B: checked-in synthetic data. Panel C: independent two-step runs, not quality gains.\n"
             "Panel D: hard-coded templates, no model inference; warnings require error analysis.\n"
             "No throughput, CTR, or held-out quality claim. Sources: docs/verification/.",
             fontsize=10, linespacing=1.5)
    fig.savefig(ROOT / "README_charts.png", dpi=160, facecolor="white",
                metadata={"Title": "ecom-posttrain artifact and execution audit", "Description":
                          "Source JSON reports and training logs in docs/verification; counts shown without smoothing."})
    plt.close(fig)


if __name__ == "__main__":
    main()
