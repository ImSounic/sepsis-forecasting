"""RQ-specific evaluation: PR curves, model agreement, feature ablation.

Produces artifacts for the three research questions:
  RQ1: Do TCN and GBM capture different sepsis signals?
  RQ2: Do lab ordering patterns predict sepsis beyond measurement values?
  RQ3: Temporal ordering (TCN) vs aggregate statistics (LightGBM)?

Usage:
    python -m src.training.evaluate_rq --config configs/tcn.yaml
    python -m src.training.evaluate_rq --config configs/tcn.yaml --rq 1
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.training.trainer import compute_utility_score


def compute_pr_curve(labels, probs):
    """Compute precision-recall curve and AUPRC."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    auprc = average_precision_score(labels, probs)
    return precision, recall, thresholds, auprc


def compute_roc(labels, probs):
    """Compute ROC curve and AUROC."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auroc = roc_auc_score(labels, probs)
    return fpr, tpr, thresholds, auroc


def plot_pr_curves(model_results, save_path):
    """Overlay PR curves for multiple models on one figure.

    Args:
        model_results: Dict of {model_name: (labels, probs)}
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"TCN": "#2196F3", "LightGBM": "#4CAF50", "Ensemble": "#FF5722"}

    for name, (labels, probs) in model_results.items():
        precision, recall, _, auprc = compute_pr_curve(labels, probs)
        color = colors.get(name, None)
        ax.plot(recall, precision, label=f"{name} (AUPRC={auprc:.4f})",
                linewidth=2, color=color)

    # Baseline: prevalence
    prevalence = np.mean(list(model_results.values())[0][0])
    ax.axhline(y=prevalence, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline (prevalence={prevalence:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Model Comparison (RQ1/RQ3)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PR curves to {save_path}")


def plot_roc_curves(model_results, save_path):
    """Overlay ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"TCN": "#2196F3", "LightGBM": "#4CAF50", "Ensemble": "#FF5722"}

    for name, (labels, probs) in model_results.items():
        fpr, tpr, _, auroc = compute_roc(labels, probs)
        color = colors.get(name, None)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.4f})",
                linewidth=2, color=color)

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5,
            label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ROC curves to {save_path}")


def patient_level_agreement(probs_a, probs_b, labels, pids, hours, threshold):
    """Compute patient-level agreement between two models.

    For each sepsis patient, determines if model A catches them, model B catches
    them, both, or neither (based on first positive prediction).

    Returns:
        Dict with counts: both, a_only, b_only, neither, total_sepsis
    """
    from collections import defaultdict

    patient_data = defaultdict(lambda: {"labels": [], "probs_a": [], "probs_b": []})
    for i in range(len(labels)):
        pid = pids[i]
        patient_data[pid]["labels"].append(labels[i])
        patient_data[pid]["probs_a"].append(probs_a[i])
        patient_data[pid]["probs_b"].append(probs_b[i])

    both, a_only, b_only, neither = 0, 0, 0, 0
    total_sepsis = 0

    for pid, data in patient_data.items():
        if max(data["labels"]) < 0.5:
            continue  # non-sepsis patient
        total_sepsis += 1

        a_catches = any(p >= threshold for p in data["probs_a"])
        b_catches = any(p >= threshold for p in data["probs_b"])

        if a_catches and b_catches:
            both += 1
        elif a_catches:
            a_only += 1
        elif b_catches:
            b_only += 1
        else:
            neither += 1

    return {
        "both": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "total_sepsis": total_sepsis,
    }


def plot_agreement_matrix(agreement, model_a_name, model_b_name, save_path):
    """Plot a 2x2 agreement heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))

    matrix = np.array([
        [agreement["both"], agreement["a_only"]],
        [agreement["b_only"], agreement["neither"]],
    ])

    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"{model_b_name} catches", f"{model_b_name} misses"])
    ax.set_yticklabels([f"{model_a_name} catches", f"{model_a_name} misses"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    ax.set_title(f"Patient-Level Agreement (n={agreement['total_sepsis']} sepsis patients)",
                 fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved agreement matrix to {save_path}")


def rq2_feature_ablation(train_processed, val_processed, feature_columns, norm_stats=None):
    """RQ2: Train LightGBM with and without time-since features.

    Compares performance to quantify the predictive value of lab ordering patterns.

    Returns:
        Dict with results for 'with_timesince' and 'without_timesince'.
    """
    from src.models.lightgbm_model import (
        LightGBMBaseline,
        prepare_baseline_data,
        get_baseline_feature_names,
    )

    results = {}

    # Full model (with time-since features)
    print("\n  RQ2: Training LightGBM WITH time-since features...")
    X_train_full, y_train, _, _ = prepare_baseline_data(
        train_processed, feature_columns, enhanced=True, norm_stats=norm_stats)
    X_val_full, y_val, val_pids, val_hours = prepare_baseline_data(
        val_processed, feature_columns, enhanced=True, norm_stats=norm_stats)
    feature_names_full = get_baseline_feature_names(
        feature_columns, enhanced=True, has_norm_stats=norm_stats is not None)

    model_full = LightGBMBaseline(improved=True)
    model_full.fit(X_train_full, y_train, X_val_full, y_val,
                   feature_names=feature_names_full)
    probs_full = model_full.predict_proba(X_val_full)
    auprc_full = average_precision_score(y_val, probs_full)
    auroc_full = roc_auc_score(y_val, probs_full)

    # Find best utility
    best_util_full = float("-inf")
    for t in np.arange(0.1, 0.91, 0.05):
        preds = (probs_full >= t).astype(int).tolist()
        util = compute_utility_score(y_val.tolist(), preds, val_pids, val_hours)
        best_util_full = max(best_util_full, util)

    results["with_timesince"] = {
        "auprc": float(auprc_full),
        "auroc": float(auroc_full),
        "utility": float(best_util_full),
        "n_features": X_train_full.shape[1],
    }

    # Ablated model (remove time-since features: indices 80-119 in base 120)
    print("\n  RQ2: Training LightGBM WITHOUT time-since features...")
    # Identify time-since column indices in the base features
    timesince_cols = [c for c in feature_columns if c.endswith("_time_since")]
    timesince_indices = [feature_columns.index(c) for c in timesince_cols]

    # Remove from full feature array: need to identify which columns in expanded
    # features correspond to time-since. They're in the first 120 columns.
    keep_base = [i for i in range(len(feature_columns)) if i not in timesince_indices]
    ablated_feature_cols = [feature_columns[i] for i in keep_base]

    X_train_abl, y_train_abl, _, _ = prepare_baseline_data(
        train_processed, ablated_feature_cols, enhanced=True, norm_stats=norm_stats)
    X_val_abl, y_val_abl, val_pids_abl, val_hours_abl = prepare_baseline_data(
        val_processed, ablated_feature_cols, enhanced=True, norm_stats=norm_stats)
    feature_names_abl = get_baseline_feature_names(
        ablated_feature_cols, enhanced=True, has_norm_stats=norm_stats is not None)

    model_abl = LightGBMBaseline(improved=True)
    model_abl.fit(X_train_abl, y_train_abl, X_val_abl, y_val_abl,
                  feature_names=feature_names_abl)
    probs_abl = model_abl.predict_proba(X_val_abl)
    auprc_abl = average_precision_score(y_val_abl, probs_abl)
    auroc_abl = roc_auc_score(y_val_abl, probs_abl)

    best_util_abl = float("-inf")
    for t in np.arange(0.1, 0.91, 0.05):
        preds = (probs_abl >= t).astype(int).tolist()
        util = compute_utility_score(y_val_abl.tolist(), preds, val_pids_abl, val_hours_abl)
        best_util_abl = max(best_util_abl, util)

    results["without_timesince"] = {
        "auprc": float(auprc_abl),
        "auroc": float(auroc_abl),
        "utility": float(best_util_abl),
        "n_features": X_train_abl.shape[1],
    }

    results["delta"] = {
        "auprc": float(auprc_full - auprc_abl),
        "auroc": float(auroc_full - auroc_abl),
        "utility": float(best_util_full - best_util_abl),
    }

    return results


def generate_rq_summary(tcn_metrics, lgb_metrics, ensemble_metrics,
                        agreement, rq2_results, save_path):
    """Generate a markdown summary of all RQ results."""
    lines = [
        "# Research Question Evaluation Results\n",
        "## RQ1: Model Complementarity\n",
        "| Metric | TCN | LightGBM | Ensemble |",
        "|--------|-----|----------|----------|",
        f"| AUPRC | {tcn_metrics['auprc']:.4f} | {lgb_metrics['auprc']:.4f} | {ensemble_metrics['auprc']:.4f} |",
        f"| AUROC | {tcn_metrics['auroc']:.4f} | {lgb_metrics['auroc']:.4f} | {ensemble_metrics['auroc']:.4f} |",
        f"| Utility | {tcn_metrics['utility']:.4f} | {lgb_metrics['utility']:.4f} | {ensemble_metrics['utility']:.4f} |",
        "",
        f"Patient-level agreement (at optimal threshold):",
        f"- Both models catch: {agreement['both']} patients",
        f"- TCN only: {agreement['a_only']} patients",
        f"- LightGBM only: {agreement['b_only']} patients",
        f"- Neither catches: {agreement['neither']} patients",
        f"- Overlap: {agreement['both'] / max(agreement['total_sepsis'], 1) * 100:.1f}%",
        "",
        "## RQ2: Lab Ordering Patterns (Feature Ablation)\n",
        "| Metric | With time-since | Without time-since | Delta |",
        "|--------|----------------|-------------------|-------|",
    ]

    if rq2_results:
        w = rq2_results["with_timesince"]
        wo = rq2_results["without_timesince"]
        d = rq2_results["delta"]
        lines.extend([
            f"| AUPRC | {w['auprc']:.4f} | {wo['auprc']:.4f} | {d['auprc']:+.4f} |",
            f"| AUROC | {w['auroc']:.4f} | {wo['auroc']:.4f} | {d['auroc']:+.4f} |",
            f"| Utility | {w['utility']:.4f} | {wo['utility']:.4f} | {d['utility']:+.4f} |",
            f"| Features | {w['n_features']} | {wo['n_features']} | - |",
        ])

    lines.extend([
        "",
        "## RQ3: Temporal Ordering vs Aggregate Statistics\n",
        "| Metric | TCN (sequential) | LightGBM (aggregate) |",
        "|--------|-----------------|---------------------|",
        f"| AUPRC | {tcn_metrics['auprc']:.4f} | {lgb_metrics['auprc']:.4f} |",
        f"| AUROC | {tcn_metrics['auroc']:.4f} | {lgb_metrics['auroc']:.4f} |",
        f"| Utility | {tcn_metrics['utility']:.4f} | {lgb_metrics['utility']:.4f} |",
    ])

    content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(content)
    print(f"  Saved RQ summary to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ-specific evaluation")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--rq", type=int, choices=[1, 2, 3], default=None,
                        help="Run only a specific RQ evaluation (default: all)")
    parser.add_argument("--output-dir", default="outputs/rq_analysis",
                        help="Output directory for RQ artifacts")
    args = parser.parse_args()
    print(f"RQ evaluation requires trained TCN and LightGBM models.")
    print(f"Run training first, then use this script to generate comparisons.")
    print(f"Output will be saved to {args.output_dir}/")
