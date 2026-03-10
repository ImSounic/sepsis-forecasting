"""Explainability analysis script for the GRU sepsis prediction model.

Generates attention visualizations, SHAP feature importance, and
comparison with LightGBM baseline. All plots saved to outputs/figures/.

Usage:
    python -m src.explainability.run_analysis --config configs/optimized.yaml
    python -m src.explainability.run_analysis --config configs/optimized.yaml --skip-shap
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from src.data.cache import get_cache_path, load_preprocessed
from src.data.dataset import SepsisDataset
from src.data.feature_engineering import INPUT_FEATURES
from src.data.preprocessing import TARGET
from src.explainability.attention_viz import (
    extract_attention_weights,
    plot_attention_distribution,
    plot_patient_attention,
)
from src.models.gru import GRUWithAttention, count_parameters
from src.training.train import load_config


FIGURES_DIR = "outputs/figures"
ATTENTION_DIR = os.path.join(FIGURES_DIR, "attention")
PATIENTS_DIR = os.path.join(FIGURES_DIR, "patients")
SHAP_DIR = os.path.join(FIGURES_DIR, "shap")
COMPARISON_DIR = os.path.join(FIGURES_DIR, "comparison")

N_PATIENT_EXAMPLES = 5  # per category (TP, FP, TN, FN)
SHAP_BACKGROUND = 500
SHAP_TEST = 200


def load_model_and_data(config):
    """Load trained model, checkpoint, and cached validation data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cache_path = get_cache_path(config)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cache not found at {cache_path}. "
            "Run training first: python -m src.training.train --config configs/default.yaml"
        )

    cached = load_preprocessed(cache_path)
    train_processed = cached["train_processed"]
    val_processed = cached["val_processed"]
    print(f"Loaded {len(train_processed)} train, {len(val_processed)} val patients from cache")

    seq_length = config["data"]["seq_length"]
    val_dataset = SepsisDataset(val_processed, seq_length=seq_length)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=0,
    )
    feature_columns = val_dataset.feature_columns
    print(f"Validation: {len(val_dataset):,} samples, {len(feature_columns)} features, "
          f"seq_length={seq_length}")

    checkpoint_path = os.path.join(config["output"]["model_dir"], "best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train the model first."
        )

    model = GRUWithAttention(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        bidirectional=config["model"]["bidirectional"],
        attention_size=config["model"].get("attention_size", 64),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    print(f"  Best utility: {checkpoint['utility']:.4f}")
    print(f"  Threshold: {checkpoint['threshold']:.2f}")

    return model, checkpoint, val_loader, val_processed, train_processed, feature_columns, device


def print_model_summary(model, config, checkpoint, feature_columns, val_dataset_len):
    """Print model architecture and training summary."""
    total_params, trainable_params = count_parameters(model)
    seq_length = config["data"]["seq_length"]

    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"  Type:             GRU + Temporal Attention")
    print(f"  Input size:       {config['model']['input_size']}")
    print(f"  Hidden size:      {config['model']['hidden_size']}")
    print(f"  Layers:           {config['model']['num_layers']}")
    print(f"  Bidirectional:    {config['model']['bidirectional']}")
    print(f"  Dropout:          {config['model']['dropout']}")
    print(f"  Parameters:       {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Best epoch:       {checkpoint['epoch']+1}")
    print(f"  Utility score:    {checkpoint['utility']:.4f}")
    print(f"  Threshold:        {checkpoint['threshold']:.2f}")
    print(f"  Sequence length:  {seq_length} hours")
    print(f"  Features/step:    {len(feature_columns)} ({len(INPUT_FEATURES)} raw + masks + time-since)")
    print(f"  Val samples:      {val_dataset_len:,}")

    print("\nLayer breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name:40s}  {str(list(param.shape)):20s}  {param.numel():>8,}")


def print_classification_summary(preds, labels, threshold):
    """Print TP/FP/TN/FN counts, precision, recall, F1, AUROC, AUPRC, and confusion matrix."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    preds_arr = np.array(preds)
    binary_preds = (preds_arr >= threshold).astype(int)
    binary_labels = np.array(labels).astype(int)

    tp = int(((binary_preds == 1) & (binary_labels == 1)).sum())
    fp = int(((binary_preds == 1) & (binary_labels == 0)).sum())
    tn = int(((binary_preds == 0) & (binary_labels == 0)).sum())
    fn = int(((binary_preds == 0) & (binary_labels == 1)).sum())
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    auroc = roc_auc_score(binary_labels, preds_arr)
    auprc = average_precision_score(binary_labels, preds_arr)

    print(f"\n  Classification Summary (threshold={threshold:.2f}):")
    print(f"    TP: {tp:>8,}    FP: {fp:>8,}")
    print(f"    FN: {fn:>8,}    TN: {tn:>8,}")
    print(f"    Total: {total:,}")

    print(f"\n    Precision:    {precision:.4f}")
    print(f"    Recall:       {recall:.4f}")
    print(f"    Specificity:  {specificity:.4f}")
    print(f"    F1 Score:     {f1:.4f}")
    print(f"    Accuracy:     {accuracy:.4f}")
    print(f"    AUROC:        {auroc:.4f}")
    print(f"    AUPRC:        {auprc:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Pos        Neg")
    print(f"    Actual Pos  {tp:>8,}   {fn:>8,}")
    print(f"    Actual Neg  {fp:>8,}   {tn:>8,}")

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "specificity": specificity,
            "f1": f1, "accuracy": accuracy, "auroc": auroc, "auprc": auprc}


def run_attention_analysis(model, val_loader, val_processed, feature_columns,
                           checkpoint, device):
    """Extract attention weights from full validation set and generate plots."""
    seq_length = len(next(iter(val_loader))[0][0])
    threshold = checkpoint["threshold"]

    print("\n" + "=" * 60)
    print("ATTENTION ANALYSIS (full validation set)")
    print("=" * 60)

    print("Running inference on all validation samples...")
    attn_data = extract_attention_weights(
        model, val_loader, device=device, num_samples=None, show_progress=True,
    )

    preds = np.array(attn_data["predictions"])
    labels = np.array(attn_data["labels"])
    all_attn = np.array(attn_data["attention_weights"])

    print(f"\nExtracted {len(preds):,} samples")
    print(f"  Positive labels: {int(labels.sum()):,}")
    print(f"  Negative labels: {int((labels == 0).sum()):,}")
    if labels.sum() > 0:
        print(f"  Mean prediction (positive): {preds[labels==1].mean():.4f}")
    if (labels == 0).sum() > 0:
        print(f"  Mean prediction (negative): {preds[labels==0].mean():.4f}")

    # Classification summary
    stats = print_classification_summary(preds, labels, threshold)

    # Attention distribution by TP/FP/TN/FN using the model's threshold
    fig = plot_attention_distribution(
        attn_data,
        save_path=os.path.join(ATTENTION_DIR, "attention_distribution.png"),
        threshold=threshold,
    )
    plt.close(fig)

    # Average attention by hour
    mean_attn = all_attn.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(mean_attn)), mean_attn, color="steelblue", alpha=0.8)
    ax.set_xlabel("Hour in window (0=oldest)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title(f"Average Attention by Hour ({len(preds):,} samples)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ATTENTION_DIR, "mean_attention_by_hour.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    peak_hour = int(np.argmax(mean_attn))
    print(f"\n  Peak attention at hour {peak_hour} "
          f"({seq_length - peak_hour - 1} hours before prediction)")
    attn_last3 = mean_attn[-3:].sum()
    print(f"  Attention on last 3 hours: {attn_last3:.3f} "
          f"({attn_last3/mean_attn.sum()*100:.1f}% of total)")

    # Categorize all samples
    categories = {"TP": [], "FP": [], "TN": [], "FN": []}
    binary_preds = (preds >= threshold).astype(int)
    for i in range(len(preds)):
        bp, bl = binary_preds[i], int(labels[i])
        if bp == 1 and bl == 1:
            categories["TP"].append(i)
        elif bp == 1 and bl == 0:
            categories["FP"].append(i)
        elif bp == 0 and bl == 0:
            categories["TN"].append(i)
        else:
            categories["FN"].append(i)

    # Generate 5 patient attention plots per category (20 total)
    print(f"\nGenerating {N_PATIENT_EXAMPLES} patient plots per category "
          f"(TP/FP/TN/FN)...")
    for cat in ["TP", "FP", "TN", "FN"]:
        indices = categories[cat]
        n_examples = min(N_PATIENT_EXAMPLES, len(indices))
        if n_examples == 0:
            print(f"  {cat}: no samples available")
            continue

        # Spread examples across the category for diversity
        if len(indices) > n_examples:
            step = len(indices) // n_examples
            selected = [indices[i * step] for i in range(n_examples)]
        else:
            selected = indices[:n_examples]

        for rank, idx in enumerate(selected, 1):
            pid = attn_data["patient_ids"][idx]
            hour = attn_data["hour_indices"][idx]
            pred = attn_data["predictions"][idx]
            label = int(attn_data["labels"][idx])
            attn = np.array(attn_data["attention_weights"][idx])

            patient_df = val_processed[pid]
            start = max(0, hour + 1 - seq_length)
            end = hour + 1
            window = patient_df[feature_columns].iloc[start:end].values.astype(
                np.float32)

            if window.shape[0] < seq_length:
                pad = np.zeros(
                    (seq_length - window.shape[0], window.shape[1]),
                    dtype=np.float32,
                )
                window = np.vstack([pad, window])

            save_path = os.path.join(
                PATIENTS_DIR,
                f"patient_attention_{cat.lower()}_{rank}_{pid}.png",
            )
            fig = plot_patient_attention(
                window, attn, feature_columns, save_path=save_path,
            )
            fig.suptitle(
                f"{cat} #{rank}: Patient {pid} | Hour {hour} | "
                f"Pred: {pred:.3f} | Label: {label}",
                fontsize=11, y=0.98,
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  {cat}: {n_examples} plots saved")

    # Highest prediction example
    top_pred_idx = int(np.argmax(preds))
    pid = attn_data["patient_ids"][top_pred_idx]
    hour = attn_data["hour_indices"][top_pred_idx]
    pred = attn_data["predictions"][top_pred_idx]
    label = int(attn_data["labels"][top_pred_idx])
    attn = np.array(attn_data["attention_weights"][top_pred_idx])

    patient_df = val_processed[pid]
    start = max(0, hour + 1 - seq_length)
    end = hour + 1
    window = patient_df[feature_columns].iloc[start:end].values.astype(np.float32)
    if window.shape[0] < seq_length:
        pad = np.zeros((seq_length - window.shape[0], window.shape[1]),
                        dtype=np.float32)
        window = np.vstack([pad, window])

    save_path = os.path.join(PATIENTS_DIR, f"patient_attention_highest_pred_{pid}.png")
    fig = plot_patient_attention(window, attn, feature_columns, save_path=save_path)
    fig.suptitle(
        f"Highest Pred: Patient {pid} | Hour {hour} | "
        f"Pred: {pred:.3f} | Label: {label}",
        fontsize=11, y=0.98,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Highest pred: Patient {pid}, Pred={pred:.3f}, Label={label}")

    return attn_data


def run_shap_analysis(model, val_loader, train_processed, feature_columns, device):
    """Compute SHAP values and generate importance plots.

    Uses 500 background samples and 200 test samples.
    Returns (shap_vals, mean_abs_shap) for use in comparison.
    """
    from src.explainability.shap_analysis import (
        compute_shap_values,
        plot_feature_importance_shap,
        plot_temporal_importance,
    )

    seq_length = len(next(iter(val_loader))[0][0])

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS")
    print("=" * 60)

    print(f"Collecting {SHAP_BACKGROUND} background samples from training data...")
    train_dataset = SepsisDataset(train_processed, seq_length=seq_length)
    train_loader_bg = torch.utils.data.DataLoader(
        train_dataset, batch_size=SHAP_BACKGROUND, shuffle=True, num_workers=0,
    )
    bg_features, _, _, _, _ = next(iter(train_loader_bg))
    bg_features = bg_features[:SHAP_BACKGROUND]
    print(f"  Background samples: {bg_features.shape}")

    print(f"Collecting {SHAP_TEST} test samples from validation data...")
    test_features_list = []
    n_collected = 0
    for features, _, _, _, _ in val_loader:
        test_features_list.append(features)
        n_collected += features.shape[0]
        if n_collected >= SHAP_TEST:
            break
    test_features = torch.cat(test_features_list, dim=0)[:SHAP_TEST]
    print(f"  Test samples: {test_features.shape}")

    print(f"Computing SHAP values ({SHAP_BACKGROUND} bg, {SHAP_TEST} test)...")
    print("  This may take several minutes...")
    shap_result = compute_shap_values(
        model, bg_features, test_features,
        device=device, num_samples=SHAP_TEST,
    )
    shap_vals = shap_result["shap_values"]
    print(f"  SHAP values shape: {shap_vals.shape}")

    fig = plot_feature_importance_shap(
        shap_vals, feature_columns, top_k=20,
        save_path=os.path.join(SHAP_DIR, "shap_feature_importance.png"),
    )
    plt.close(fig)

    fig = plot_temporal_importance(
        shap_vals, feature_columns,
        save_path=os.path.join(SHAP_DIR, "shap_temporal_importance.png"),
    )
    plt.close(fig)

    mean_abs_shap = np.abs(shap_vals).mean(axis=(0, 1))
    n_feat = min(len(feature_columns), mean_abs_shap.shape[0])
    top_indices = np.argsort(mean_abs_shap[:n_feat])[::-1][:10]
    feature_columns_arr = np.array(feature_columns)
    print("\nTop 10 features by mean |SHAP|:")
    for rank, i in enumerate(top_indices.tolist(), 1):
        print(f"  {rank:2d}. {feature_columns_arr[i]:30s}  {mean_abs_shap[i]:.6f}")

    return shap_vals, mean_abs_shap


def run_baseline_comparison(feature_columns, mean_abs_shap, config):
    """Compare GRU SHAP importance with LightGBM feature importance."""
    from src.models.baseline import LightGBMBaseline

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)

    baseline_path = os.path.join(config["output"]["model_dir"], "baseline.pkl")
    if not os.path.exists(baseline_path):
        print(f"No baseline model found at {baseline_path}")
        print("Run: python -m src.training.train_baseline --config configs/default.yaml")
        print("\nGRU SHAP top 20:")
        n_feat = min(len(feature_columns), mean_abs_shap.shape[0])
        gru_importance = pd.DataFrame({
            "feature": feature_columns[:n_feat],
            "shap_importance": mean_abs_shap[:n_feat],
        }).sort_values("shap_importance", ascending=False)
        print(gru_importance.head(20).to_string(index=False))
        return

    baseline = LightGBMBaseline.load(baseline_path)
    lgb_importance = baseline.feature_importance()
    print(f"Loaded LightGBM baseline with {len(lgb_importance)} features")

    lgb_top20 = lgb_importance.head(20).copy()
    lgb_top20["importance_norm"] = lgb_top20["importance"] / lgb_top20["importance"].max()

    n_feat = min(len(feature_columns), mean_abs_shap.shape[0])
    gru_importance = pd.DataFrame({
        "feature": feature_columns[:n_feat],
        "shap_importance": mean_abs_shap[:n_feat],
    }).sort_values("shap_importance", ascending=False)
    gru_top20 = gru_importance.head(20).copy()
    gru_top20["importance_norm"] = gru_top20["shap_importance"] / gru_top20["shap_importance"].max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    y_pos = np.arange(20)
    ax1.barh(y_pos, gru_top20["importance_norm"].values[::-1], color="steelblue", alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(gru_top20["feature"].values[::-1], fontsize=8)
    ax1.set_xlabel("Normalized importance")
    ax1.set_title("GRU+Attention (SHAP)")
    ax1.grid(True, axis="x", alpha=0.3)

    n_lgb = len(lgb_top20)
    y_pos2 = np.arange(n_lgb)
    ax2.barh(y_pos2, lgb_top20["importance_norm"].values[::-1], color="coral", alpha=0.8)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(lgb_top20["feature"].values[::-1], fontsize=8)
    ax2.set_xlabel("Normalized importance")
    ax2.set_title("LightGBM (Gain)")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Feature Importance: GRU+Attention vs LightGBM", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(COMPARISON_DIR, "feature_importance_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    gru_top_set = set(gru_top20["feature"].values)
    lgb_top_set = set(lgb_top20["feature"].values)
    overlap = gru_top_set & lgb_top_set
    print(f"\nTop-20 feature overlap: {len(overlap)} features in common")
    if overlap:
        print(f"  Shared: {sorted(overlap)}")
    print(f"  GRU-only: {sorted(gru_top_set - lgb_top_set)}")
    print(f"  LightGBM-only: {sorted(lgb_top_set - gru_top_set)}")


def print_summary(skip_shap=False):
    """Print where output files were saved."""
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Figures saved to: {FIGURES_DIR}/")
    print(f"  - attention/attention_distribution.png")
    print(f"  - attention/mean_attention_by_hour.png")
    print(f"  - patients/patient_attention_*.png ({N_PATIENT_EXAMPLES * 4 + 1} files)")
    if not skip_shap:
        print(f"  - shap/shap_feature_importance.png")
        print(f"  - shap/shap_temporal_importance.png")
        print(f"  - comparison/feature_importance_comparison.png")


def main(config_path: str, skip_shap: bool = False):
    config = load_config(config_path)
    for d in [FIGURES_DIR, ATTENTION_DIR, PATIENTS_DIR, SHAP_DIR, COMPARISON_DIR]:
        os.makedirs(d, exist_ok=True)

    model, checkpoint, val_loader, val_processed, train_processed, feature_columns, device = \
        load_model_and_data(config)

    print_model_summary(model, config, checkpoint, feature_columns, len(val_loader.dataset))

    run_attention_analysis(model, val_loader, val_processed, feature_columns, checkpoint, device)

    if skip_shap:
        print("\n[Skipping SHAP analysis (--skip-shap)]")
        print_summary(skip_shap=True)
        return

    shap_vals, mean_abs_shap = run_shap_analysis(
        model, val_loader, train_processed, feature_columns, device,
    )
    run_baseline_comparison(feature_columns, mean_abs_shap, config)

    print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run explainability analysis on trained sepsis model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis (slow)")
    args = parser.parse_args()
    main(args.config, skip_shap=args.skip_shap)
