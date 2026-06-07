"""Per-class accuracy comparison plot for the four trained models.

Reads test_metrics.json from each checkpoint directory and produces a
grouped bar chart at outputs/figures/per_class_comparison.png.

This is the headline figure for the project README.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CHECKPOINTS = Path("checkpoints")
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Order matters for the legend: simplest model first, most complex last
MODELS = [
    ("s1_only", "S1-only", "#7B7D7D"),       # gray  - SAR alone
    ("s2_only", "S2-only", "#2E86AB"),       # blue  - optical alone
    ("early_fusion", "Early fusion", "#F18F01"),  # orange
    ("late_fusion", "Late fusion", "#A23B72"),    # purple - best
]

CLASS_ORDER = [
    "Forest", "Shrubland", "Grassland", "Wetlands",
    "Croplands", "Urban", "Barren", "Water",
]


def load_per_class_accuracy(name: str) -> dict[str, float]:
    """Load per-class test recall for one model."""
    path = CHECKPOINTS / name / "test_metrics.json"
    with open(path) as f:
        data = json.load(f)
    return data["test_per_class_accuracy"]


def main() -> None:
    # Load all four sets of per-class accuracies
    model_data = {name: load_per_class_accuracy(name) for name, _, _ in MODELS}

    # Build the arrays for plotting: shape (n_models, n_classes)
    n_classes = len(CLASS_ORDER)
    n_models = len(MODELS)
    values = np.array([
        [model_data[name][cls] for cls in CLASS_ORDER]
        for name, _, _ in MODELS
    ])

    # Grouped bar chart
    bar_width = 0.2
    x = np.arange(n_classes)
    offsets = np.linspace(
        -(n_models - 1) / 2 * bar_width,
        (n_models - 1) / 2 * bar_width,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, ((_, label, color), offset) in enumerate(zip(MODELS, offsets)):
        bars = ax.bar(
            x + offset,
            values[i],
            width=bar_width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Test-set per-class recall  (TP / (TP + FN))")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Per-class recall on the DFC2020 test set (5,128 patches, held-out region)",
        loc="left",
        fontsize=12,
        pad=10,
    )

    # Grid for readability; only horizontal lines
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend below the title to avoid cluttering bars
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
    )

    plt.tight_layout()
    out_path = OUT_DIR / "per_class_comparison.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()