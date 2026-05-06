"""
=============================================================
  Evaluation & Visualization Utilities
=============================================================
Handles:
  • Generating a full classification report
  • Plotting the confusion matrix
  • Plotting training loss / metric curves
  • Per-class F1 bar chart
"""

import os
import logging
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

logger = logging.getLogger(__name__)

# Consistent color palette
PALETTE = sns.color_palette("mako", as_cmap=False)


# ──────────────────────────────────────────────────────────────
# 1. Classification Report
# ──────────────────────────────────────────────────────────────

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: Dict[int, str],
) -> None:
    """Print a detailed per-class precision / recall / F1 table."""
    target_names = [id2label[i] for i in sorted(id2label)]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\n" + "=" * 60)
    print("  Classification Report")
    print("=" * 60)
    print(report)
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────
# 2. Confusion Matrix
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: Dict[int, str],
    save_path: str = "./confusion_matrix.png",
) -> None:
    """Save a styled confusion matrix heatmap."""
    labels     = sorted(id2label)
    label_names = [id2label[i] for i in labels]
    cm         = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm    = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) - 1)))
    sns.heatmap(
        cm_norm,
        annot=cm,          # show raw counts
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        linewidths=0.5,
        linecolor="grey",
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix – RoBERTa + LoRA on LUCID", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


# ──────────────────────────────────────────────────────────────
# 3. Training Curves
# ──────────────────────────────────────────────────────────────

def plot_training_curves(
    log_history: List[Dict],
    save_path: str = "./training_curves.png",
) -> None:
    """
    Plot training loss, validation loss, and validation F1 across epochs.

    Parameters
    ----------
    log_history : trainer.state.log_history  (list of dicts)
    save_path   : where to save the figure
    """
    train_steps, train_losses = [], []
    eval_steps, eval_losses, eval_f1s = [], [], []

    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])
        if "eval_f1" in entry:
            eval_f1s.append(entry["eval_f1"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss curves ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(train_steps, train_losses, label="Train Loss",
            color="#4C72B0", linewidth=2, alpha=0.85)
    if eval_steps:
        ax.plot(eval_steps, eval_losses, label="Val Loss",
                color="#DD8452", linewidth=2, linestyle="--", alpha=0.85)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss",  fontsize=11)
    ax.set_title("Training & Validation Loss", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── F1 curve ────────────────────────────────────────────
    ax = axes[1]
    if eval_f1s:
        ax.plot(eval_steps[:len(eval_f1s)], eval_f1s, label="Val Macro-F1",
                color="#55A868", linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("Step",     fontsize=11)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title("Validation Macro-F1 over Training", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle("RoBERTa + PEFT LoRA  –  LUCID Dialogue Act Classification",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved → {save_path}")


# ──────────────────────────────────────────────────────────────
# 4. Per-Class F1 Bar Chart
# ──────────────────────────────────────────────────────────────

def plot_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: Dict[int, str],
    save_path: str = "./per_class_f1.png",
) -> None:
    """Horizontal bar chart showing per-class F1 score."""
    from sklearn.metrics import f1_score

    labels      = sorted(id2label)
    label_names = [id2label[i] for i in labels]
    f1_scores   = f1_score(y_true, y_pred, labels=labels, average=None)

    # Sort descending
    order       = np.argsort(f1_scores)[::-1]
    sorted_names  = [label_names[i] for i in order]
    sorted_scores = f1_scores[order]

    colors = plt.cm.RdYlGn(sorted_scores)

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
    bars = ax.barh(sorted_names, sorted_scores, color=colors, edgecolor="white")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title("Per-Class F1 Score – LUCID Dialogue Acts", fontsize=13)
    ax.axvline(x=np.mean(f1_scores), color="navy", linestyle="--",
               linewidth=1.2, label=f"Macro avg = {np.mean(f1_scores):.3f}")
    ax.legend(fontsize=9)

    for bar, score in zip(bars, sorted_scores):
        ax.text(
            score + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", ha="left", fontsize=9
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Per-class F1 chart saved → {save_path}")
