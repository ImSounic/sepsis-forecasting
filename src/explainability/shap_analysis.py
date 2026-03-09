"""SHAP-based feature importance analysis for the GRU sepsis model."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger("sepsis")


class _ModelWrapper(torch.nn.Module):
    """Wraps GRUWithAttention to return only logits (no attention weights).

    SHAP explainers expect a single output tensor from forward().
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits.unsqueeze(-1)


def compute_shap_values(model, background_data, test_data, device="cpu", num_samples=100):
    """Compute SHAP values using GradientExplainer.

    cuDNN's RNN backward pass only works in training mode, so we set
    model.train() but disable dropout to keep inference-like behavior.
    If GradientExplainer still fails (e.g. incompatible SHAP version),
    falls back to DeepExplainer.

    Args:
        model: Trained GRUWithAttention model.
        background_data: Tensor (n_background, seq_len, n_features) for baseline.
        test_data: Tensor (n_test, seq_len, n_features) to explain.
        device: Torch device string.
        num_samples: Max test samples to explain.

    Returns:
        Dict with keys:
            shap_values: numpy array (num_samples, seq_len, n_features)
            test_data: numpy array of the explained inputs
    """
    import shap

    model.to(device)
    wrapper = _ModelWrapper(model).to(device)

    background = background_data[:200].to(device)
    test = test_data[:num_samples].to(device)

    # cuDNN RNN backward requires training mode. Disable dropout layers
    # so behavior stays inference-like despite model.train().
    saved_dropout_states = {}
    for name, module in wrapper.named_modules():
        if isinstance(module, torch.nn.Dropout):
            saved_dropout_states[name] = module.p
            module.p = 0.0

    # Disable cuDNN so the non-cuDNN RNN backward works in eval mode too
    cudnn_was_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    try:
        wrapper.train()
        explainer = shap.GradientExplainer(wrapper, background)
        shap_vals = explainer.shap_values(test)
    except RuntimeError as e:
        logger.warning("GradientExplainer failed (%s), trying DeepExplainer", e)
        wrapper.train()
        explainer = shap.DeepExplainer(wrapper, background)
        shap_vals = explainer.shap_values(test)
    finally:
        torch.backends.cudnn.enabled = cudnn_was_enabled
        wrapper.eval()
        for name, module in wrapper.named_modules():
            if name in saved_dropout_states:
                module.p = saved_dropout_states[name]

    # GradientExplainer returns a list for multi-output; we have single output
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    shap_vals = np.array(shap_vals)
    # Squeeze trailing dim from wrapper's unsqueeze(-1) -> (n, seq, features)
    if shap_vals.ndim == 4:
        shap_vals = shap_vals.squeeze(-1)

    result = {
        "shap_values": shap_vals,
        "test_data": test.cpu().numpy(),
    }
    logger.info("Computed SHAP values for %d samples", result["shap_values"].shape[0])
    return result


def plot_feature_importance_shap(shap_values, feature_names, top_k=20, save_path=None):
    """Plot mean absolute SHAP value per feature.

    Aggregates across all timesteps and samples to show which features
    matter most overall.

    Args:
        shap_values: numpy array (n_samples, seq_len, n_features).
        feature_names: List of feature names matching last dimension.
        top_k: Number of top features to display.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    # Mean |SHAP| across samples and timesteps -> (n_features,)
    mean_abs = np.abs(shap_values).mean(axis=(0, 1))

    # Sort and take top_k
    n_features = min(len(feature_names), mean_abs.shape[0])
    indices = np.argsort(mean_abs[:n_features])[::-1][:top_k]
    feature_names = np.array(feature_names)
    top_names = feature_names[indices].tolist()
    top_values = mean_abs[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.35)))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_values[::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title("Feature Importance (SHAP)", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved SHAP feature importance plot to %s", save_path)

    return fig


def plot_temporal_importance(shap_values, feature_names=None, save_path=None):
    """Show which timesteps matter most, aggregated across features.

    Args:
        shap_values: numpy array (n_samples, seq_len, n_features).
        feature_names: Optional, not used directly but kept for API consistency.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    # Mean |SHAP| across samples and features -> (seq_len,)
    temporal = np.abs(shap_values).mean(axis=(0, 2))
    hours = np.arange(len(temporal))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hours, temporal, color="coral", alpha=0.8, width=0.8)
    ax.set_xlabel("Hour in window (0 = oldest)", fontsize=10)
    ax.set_ylabel("Mean |SHAP value|", fontsize=10)
    ax.set_title("Temporal Importance (SHAP)", fontsize=11)
    ax.set_xlim(-0.5, len(temporal) - 0.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved temporal importance plot to %s", save_path)

    return fig
