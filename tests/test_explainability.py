"""Minimal smoke tests for explainability modules."""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.gru import GRUWithAttention


# Shared fixtures

@pytest.fixture
def small_model():
    """Small GRU model for testing (input_size=12, fast)."""
    return GRUWithAttention(input_size=12, hidden_size=16, num_layers=1,
                            dropout=0.0, bidirectional=False, attention_size=8)


@pytest.fixture
def dummy_dataloader():
    """DataLoader mimicking SepsisDataset output format.

    Yields (features, labels, masks, pids, hours) per batch.
    """
    n_samples, seq_len, n_features = 20, 8, 12
    features = torch.randn(n_samples, seq_len, n_features)
    labels = torch.zeros(n_samples)
    labels[:5] = 1.0  # 5 positive samples
    masks = torch.ones(n_samples, seq_len)
    pids = [f"p{i:04d}" for i in range(n_samples)]
    hours = list(range(n_samples))

    class _FakeDataset:
        def __init__(self):
            self.feature_columns = [f"feat_{i}" for i in range(n_features)]

    class _FakeLoader:
        """Mimics DataLoader iteration returning 5-tuples."""
        def __init__(self):
            self.dataset = _FakeDataset()
            self.batch_size = 8

        def __iter__(self):
            for start in range(0, n_samples, 8):
                end = min(start + 8, n_samples)
                yield (
                    features[start:end],
                    labels[start:end],
                    masks[start:end],
                    pids[start:end],
                    hours[start:end],
                )

    return _FakeLoader()


class TestAttentionViz:
    """Tests for src.explainability.attention_viz."""

    def test_extract_attention_weights(self, small_model, dummy_dataloader):
        from src.explainability.attention_viz import extract_attention_weights

        result = extract_attention_weights(small_model, dummy_dataloader,
                                           device="cpu", num_samples=10)

        assert len(result["patient_ids"]) == 10
        assert len(result["predictions"]) == 10
        assert len(result["labels"]) == 10
        assert len(result["attention_weights"]) == 10
        assert len(result["hour_indices"]) == 10

        # Attention weights should be valid probabilities
        for attn in result["attention_weights"]:
            arr = np.array(attn)
            assert arr.min() >= 0.0
            assert abs(arr.sum() - 1.0) < 1e-4

    def test_extract_respects_num_samples(self, small_model, dummy_dataloader):
        from src.explainability.attention_viz import extract_attention_weights

        result = extract_attention_weights(small_model, dummy_dataloader,
                                           device="cpu", num_samples=5)
        assert len(result["patient_ids"]) == 5

    def test_plot_patient_attention(self):
        from src.explainability.attention_viz import plot_patient_attention

        seq_len, n_features = 8, 12
        patient_data = np.random.randn(seq_len, n_features)
        attn = np.random.dirichlet(np.ones(seq_len))
        # Use feature names that include some vital signs
        names = ["HR", "O2Sat", "Temp", "SBP"] + [f"feat_{i}" for i in range(8)]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_patient.png")
            fig = plot_patient_attention(patient_data, attn, names, save_path=save_path)

            assert isinstance(fig, plt.Figure)
            assert os.path.exists(save_path)
            plt.close(fig)

    def test_plot_attention_distribution(self):
        from src.explainability.attention_viz import plot_attention_distribution

        data = {
            "attention_weights": [np.random.dirichlet(np.ones(8)).tolist() for _ in range(20)],
            "predictions": [0.8] * 5 + [0.7] * 5 + [0.1] * 5 + [0.2] * 5,
            "labels": [1.0] * 5 + [0.0] * 5 + [0.0] * 5 + [1.0] * 5,
        }

        fig = plot_attention_distribution(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_generate_attention_report(self, small_model, dummy_dataloader):
        from src.explainability.attention_viz import generate_attention_report

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_attention_report(
                small_model, dummy_dataloader, tmpdir,
                device="cpu", num_samples=10,
            )
            assert os.path.exists(report_path)
            assert report_path.endswith(".html")

            with open(report_path) as f:
                html = f.read()
            assert "Attention Analysis Report" in html
            assert "attention_distribution.png" in html


def _shap_available():
    try:
        import shap
        return True
    except ImportError:
        return False


class TestShapAnalysis:
    """Tests for src.explainability.shap_analysis."""

    def test_plot_feature_importance_shap(self):
        from src.explainability.shap_analysis import plot_feature_importance_shap

        shap_vals = np.random.randn(10, 8, 12)
        names = [f"feat_{i}" for i in range(12)]

        fig = plot_feature_importance_shap(shap_vals, names, top_k=5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_temporal_importance(self):
        from src.explainability.shap_analysis import plot_temporal_importance

        shap_vals = np.random.randn(10, 8, 12)

        fig = plot_temporal_importance(shap_vals)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_feature_importance_with_save(self):
        from src.explainability.shap_analysis import plot_feature_importance_shap

        shap_vals = np.random.randn(10, 8, 12)
        names = [f"feat_{i}" for i in range(12)]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "shap_importance.png")
            fig = plot_feature_importance_shap(shap_vals, names, save_path=save_path)
            assert os.path.exists(save_path)
            plt.close(fig)

    def test_plot_temporal_importance_with_save(self):
        from src.explainability.shap_analysis import plot_temporal_importance

        shap_vals = np.random.randn(10, 8, 12)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "temporal.png")
            fig = plot_temporal_importance(shap_vals, save_path=save_path)
            assert os.path.exists(save_path)
            plt.close(fig)

    @pytest.mark.skipif(
        not _shap_available(),
        reason="shap not installed",
    )
    def test_compute_shap_values(self, small_model):
        from src.explainability.shap_analysis import compute_shap_values

        bg = torch.randn(50, 8, 12)
        test = torch.randn(10, 8, 12)

        result = compute_shap_values(small_model, bg, test, device="cpu", num_samples=5)

        assert "shap_values" in result
        assert "test_data" in result
        assert result["shap_values"].shape[0] == 5
        assert result["shap_values"].shape[2] == 12
