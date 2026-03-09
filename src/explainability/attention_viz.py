"""Attention weight visualization for the GRU sepsis prediction model."""

import logging
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.preprocessing import VITAL_SIGNS

logger = logging.getLogger("sepsis")


def extract_attention_weights(model, dataloader, device="cpu", num_samples=100):
    """Run model on samples and collect attention weights.

    Args:
        model: Trained GRUWithAttention model.
        dataloader: DataLoader yielding (features, labels, masks, pids, hours).
        device: Torch device string.
        num_samples: Max samples to collect.

    Returns:
        Dict with keys: patient_ids, hour_indices, attention_weights,
        predictions, labels. Each is a list of length <= num_samples.
    """
    model.eval()
    model.to(device)

    result = {
        "patient_ids": [],
        "hour_indices": [],
        "attention_weights": [],
        "predictions": [],
        "labels": [],
    }
    collected = 0

    with torch.no_grad():
        for features, labels, masks, pids, hours in dataloader:
            if collected >= num_samples:
                break

            features = features.to(device)
            masks = masks.to(device)
            probs, attn = model.predict_proba(features, padding_mask=masks)

            batch_size = features.size(0)
            remaining = num_samples - collected
            n = min(batch_size, remaining)

            result["patient_ids"].extend(list(pids[:n]))
            result["hour_indices"].extend([int(h) for h in hours[:n]])
            result["attention_weights"].extend(attn[:n].cpu().numpy().tolist())
            result["predictions"].extend(probs[:n].cpu().numpy().tolist())
            result["labels"].extend(labels[:n].numpy().tolist())

            collected += n

    logger.info("Extracted attention weights for %d samples", collected)
    return result


def plot_patient_attention(patient_data, attention_weights, feature_names, save_path=None):
    """Plot vital signs over time with attention heatmap overlay.

    Args:
        patient_data: numpy array (seq_len, num_features) for one sample window.
        attention_weights: numpy array (seq_len,) of attention weights.
        feature_names: List of feature names matching columns of patient_data.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    seq_len = patient_data.shape[0]
    hours = np.arange(seq_len)
    attn = np.array(attention_weights)

    # Select vital signs that exist in the feature set
    vital_indices = []
    vital_names = []
    for v in VITAL_SIGNS:
        if v in feature_names:
            vital_indices.append(feature_names.index(v))
            vital_names.append(v)

    n_vitals = len(vital_names)
    if n_vitals == 0:
        n_vitals = 1  # fallback: just show attention

    fig, axes = plt.subplots(n_vitals + 1, 1, figsize=(12, 2.5 * (n_vitals + 1)),
                             sharex=True, gridspec_kw={"height_ratios": [1] * n_vitals + [0.5]})
    if n_vitals + 1 == 1:
        axes = [axes]

    # Plot each vital sign with attention shading
    for i, (idx, name) in enumerate(zip(vital_indices, vital_names)):
        ax = axes[i]
        values = patient_data[:, idx]
        ax.plot(hours, values, "b-", linewidth=1.5, label=name)

        # Shade by attention weight
        for t in range(seq_len):
            ax.axvspan(t - 0.5, t + 0.5, alpha=attn[t] * 0.8, color="red")

        ax.set_ylabel(name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # Attention bar chart at bottom
    ax_attn = axes[-1]
    ax_attn.bar(hours, attn, color="red", alpha=0.7, width=0.8)
    ax_attn.set_ylabel("Attention", fontsize=9)
    ax_attn.set_xlabel("Hour in window", fontsize=9)
    ax_attn.set_xlim(-0.5, seq_len - 0.5)
    ax_attn.tick_params(labelsize=8)

    fig.suptitle("Patient Vital Signs with Attention Weights", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved patient attention plot to %s", save_path)

    return fig


def plot_attention_distribution(attention_data, save_path=None):
    """Aggregate attention patterns across patients by outcome category.

    Shows average attention by hour-before-prediction for TP, FP, TN, FN.

    Args:
        attention_data: Dict from extract_attention_weights().
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    categories = {"TP": [], "FP": [], "TN": [], "FN": []}

    for attn, pred, label in zip(
        attention_data["attention_weights"],
        attention_data["predictions"],
        attention_data["labels"],
    ):
        binary_pred = 1 if pred >= 0.5 else 0
        binary_label = int(label)

        if binary_pred == 1 and binary_label == 1:
            categories["TP"].append(attn)
        elif binary_pred == 1 and binary_label == 0:
            categories["FP"].append(attn)
        elif binary_pred == 0 and binary_label == 0:
            categories["TN"].append(attn)
        else:
            categories["FN"].append(attn)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = {"TP": "green", "FP": "orange", "TN": "blue", "FN": "red"}

    for ax, (cat, arrays) in zip(axes, categories.items()):
        if arrays:
            stacked = np.array(arrays)
            mean_attn = stacked.mean(axis=0)
            std_attn = stacked.std(axis=0)
            hours = np.arange(len(mean_attn))

            ax.plot(hours, mean_attn, color=colors[cat], linewidth=2)
            ax.fill_between(hours, mean_attn - std_attn, mean_attn + std_attn,
                            color=colors[cat], alpha=0.2)
            ax.set_title(f"{cat} (n={len(arrays)})", fontsize=10)
        else:
            ax.set_title(f"{cat} (n=0)", fontsize=10)
            ax.text(0.5, 0.5, "No samples", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

        ax.set_xlabel("Hour in window", fontsize=9)
        ax.set_ylabel("Mean attention weight", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    fig.suptitle("Attention Distribution by Prediction Outcome", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved attention distribution plot to %s", save_path)

    return fig


def generate_attention_report(model, dataloader, output_dir, device="cpu", num_samples=50):
    """Generate an HTML report with attention visualizations.

    Creates individual patient plots and summary statistics, bundled into
    a single HTML file.

    Args:
        model: Trained GRUWithAttention model.
        dataloader: DataLoader for evaluation data.
        output_dir: Directory for output files (HTML + images).
        device: Torch device string.
        num_samples: Number of samples to include.

    Returns:
        Path to the generated HTML report.
    """
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Extract attention data
    data = extract_attention_weights(model, dataloader, device=device, num_samples=num_samples)

    # Get feature names from dataloader
    dataset = dataloader.dataset
    feature_columns = dataset.feature_columns

    # Generate distribution plot
    dist_path = os.path.join(img_dir, "attention_distribution.png")
    fig = plot_attention_distribution(data, save_path=dist_path)
    plt.close(fig)

    # Generate individual patient plots (up to 10 detailed examples)
    n_detailed = min(10, len(data["patient_ids"]))
    patient_plots = []

    for i in range(n_detailed):
        # Reconstruct the data window from the dataloader for this sample
        attn = np.array(data["attention_weights"][i])
        pid = data["patient_ids"][i]
        pred = data["predictions"][i]
        label = data["labels"][i]

        # Create a simple attention-only plot for the report
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.bar(range(len(attn)), attn, color="red", alpha=0.7)
        ax.set_xlabel("Hour in window")
        ax.set_ylabel("Attention")
        ax.set_title(f"Patient {pid} | Pred: {pred:.3f} | Label: {int(label)}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_name = f"patient_{i}_{pid}.png"
        plot_path = os.path.join(img_dir, plot_name)
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        patient_plots.append({
            "filename": f"images/{plot_name}",
            "pid": pid,
            "prediction": pred,
            "label": int(label),
        })

    # Summary statistics
    preds = np.array(data["predictions"])
    labels = np.array(data["labels"])
    all_attn = np.array(data["attention_weights"])

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    mean_pred_pos = float(preds[labels == 1].mean()) if n_pos > 0 else 0.0
    mean_pred_neg = float(preds[labels == 0].mean()) if n_neg > 0 else 0.0

    # Peak attention hour (most attended on average)
    mean_attn_by_hour = all_attn.mean(axis=0)
    peak_hour = int(np.argmax(mean_attn_by_hour))

    # Build HTML
    html_parts = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        "<title>Attention Analysis Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }",
        "h1, h2 { color: #333; } table { border-collapse: collapse; margin: 10px 0; }",
        "td, th { border: 1px solid #ccc; padding: 6px 12px; text-align: left; }",
        "th { background: #f0f0f0; } img { max-width: 100%; }",
        ".pos { color: #c00; } .neg { color: #060; }",
        "</style></head><body>",
        f"<h1>Attention Analysis Report</h1>",
        f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
        f"<p>Samples analyzed: {len(data['patient_ids'])}</p>",
        "<h2>Summary Statistics</h2>",
        "<table>",
        f"<tr><th>Metric</th><th>Value</th></tr>",
        f"<tr><td>Total samples</td><td>{len(labels)}</td></tr>",
        f"<tr><td>Positive labels</td><td>{n_pos}</td></tr>",
        f"<tr><td>Negative labels</td><td>{n_neg}</td></tr>",
        f"<tr><td>Mean prediction (positive)</td><td>{mean_pred_pos:.4f}</td></tr>",
        f"<tr><td>Mean prediction (negative)</td><td>{mean_pred_neg:.4f}</td></tr>",
        f"<tr><td>Peak attention hour</td><td>{peak_hour} (0-indexed)</td></tr>",
        "</table>",
        "<h2>Attention Distribution by Outcome</h2>",
        f"<img src='images/attention_distribution.png' alt='Attention distribution'>",
        "<h2>Individual Patient Examples</h2>",
    ]

    for p in patient_plots:
        label_class = "pos" if p["label"] == 1 else "neg"
        html_parts.append(
            f"<h3>Patient {p['pid']} "
            f"(Pred: {p['prediction']:.3f}, "
            f"Label: <span class='{label_class}'>{p['label']}</span>)</h3>"
        )
        html_parts.append(f"<img src='{p['filename']}' alt='Patient {p['pid']}'>")

    html_parts.append("</body></html>")

    report_path = os.path.join(output_dir, "attention_report.html")
    with open(report_path, "w") as f:
        f.write("\n".join(html_parts))

    logger.info("Generated attention report at %s", report_path)
    return report_path
