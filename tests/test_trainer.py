"""Tests for training loop and utility score evaluation."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SepsisDataset
from src.models.gru import GRUWithAttention
from src.training.trainer import SepsisTrainer, compute_utility_score


# Shared test patients: A (non-sepsis, 5h), B (sepsis onset hour 8, 10h)
def _build_test_scenario():
    """Return (labels, patient_ids, hours) for a 2-patient test scenario."""
    labels = [0] * 5 + [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    pids = ["A"] * 5 + ["B"] * 10
    hours = list(range(5)) + list(range(10))
    return labels, pids, hours


class TestUtilityScore:
    def test_optimal_detection(self):
        """Detect sepsis 6 hours early -> score = 1.0."""
        labels, pids, hours = _build_test_scenario()
        # Patient A: all negative.  Patient B: predict 1 at hour 2 (dt = 2-8 = -6)
        preds = [0] * 5 + [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        score = compute_utility_score(labels, preds, pids, hours)
        assert score == pytest.approx(1.0, abs=1e-3)

    def test_all_negative_is_baseline(self):
        """Predicting all 0s gives score = 0.0 (baseline)."""
        labels, pids, hours = _build_test_scenario()
        preds = [0] * 15
        score = compute_utility_score(labels, preds, pids, hours)
        assert score == pytest.approx(0.0, abs=1e-3)

    def test_early_partial_credit(self):
        """Detection 8 hours early (dt=-8, in [-12,-6)) gets partial credit."""
        labels, pids, hours = _build_test_scenario()
        preds = [0] * 5 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # hour 0 of patient B
        score = compute_utility_score(labels, preds, pids, hours)
        # credit = (−8+12)/6 = 0.667
        # normalized = (0.667+2)/(1+2) = 0.889
        assert score == pytest.approx(0.889, abs=0.01)

    def test_late_in_window(self):
        """Detection at onset time (dt=0) gets partial credit: (3-0)/9 = 1/3."""
        labels, pids, hours = _build_test_scenario()
        preds = [0] * 5 + [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # hour 8 of patient B
        score = compute_utility_score(labels, preds, pids, hours)
        # credit = (3-0)/9 = 1/3, normalized = (1/3+2)/(1+2) = 7/9 ≈ 0.778
        assert score == pytest.approx(7 / 9, abs=1e-3)

    def test_too_early_is_false_alarm(self):
        """Detection 15 hours early (dt=-15, < -12) counts as false alarm."""
        # Patient C: sepsis at hour 20, 25 hours total
        labels = [0] * 20 + [1] * 5
        pids = ["C"] * 25
        hours = list(range(25))
        preds = [0] * 5 + [1] + [0] * 19  # predict at hour 5, dt = 5-20 = -15
        score = compute_utility_score(labels, preds, pids, hours)
        # observed = -0.05, baseline = -2.0, optimal = 1.0
        # (−0.05+2)/(1+2) = 0.65
        assert score == pytest.approx(0.65, abs=0.01)

    def test_false_positives_lower_score(self):
        """False positives on non-sepsis patients reduce score."""
        labels, pids, hours = _build_test_scenario()

        # Perfect detection, no FPs
        perfect_preds = [0] * 5 + [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        perfect = compute_utility_score(labels, perfect_preds, pids, hours)

        # Same detection, but 5 FPs on patient A
        fp_preds = [1] * 5 + [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        with_fps = compute_utility_score(labels, fp_preds, pids, hours)

        assert with_fps < perfect
        assert with_fps > 0  # still positive because detection works

    def test_missed_sepsis_negative(self):
        """Missing all sepsis cases scores below baseline (negative)."""
        labels, pids, hours = _build_test_scenario()
        # Miss sepsis AND trigger false alarms on non-sepsis
        preds = [1] * 5 + [0] * 10
        score = compute_utility_score(labels, preds, pids, hours)
        assert score < 0

    def test_no_sepsis_patients(self):
        """Score is 0 when no sepsis patients exist."""
        labels = [0, 0, 0, 0, 0]
        pids = ["A"] * 5
        hours = list(range(5))
        preds = [0] * 5
        score = compute_utility_score(labels, preds, pids, hours)
        assert score == pytest.approx(0.0)


# Helpers for trainer tests
N_FEATURES = 4


def _make_patients(n_patients=10, n_hours=10, sepsis_ratio=0.2):
    """Create synthetic patient DataFrames for trainer tests."""
    patients = {}
    for i in range(n_patients):
        features = np.random.randn(n_hours, N_FEATURES).astype(np.float32)
        cols = [f"f{j}" for j in range(N_FEATURES)]
        df = pd.DataFrame(features, columns=cols)
        df["SepsisLabel"] = 0.0
        if i < int(n_patients * sepsis_ratio):
            onset = n_hours - 2
            df.loc[onset:, "SepsisLabel"] = 1.0
        patients[f"p{i}"] = df
    return patients


def _make_trainer(patients=None, config_overrides=None):
    """Create a SepsisTrainer with tiny model and synthetic data."""
    if patients is None:
        patients = _make_patients()

    dataset = SepsisDataset(patients, seq_length=4)
    loader = DataLoader(dataset, batch_size=32, num_workers=0)

    model = GRUWithAttention(
        input_size=N_FEATURES, hidden_size=8, num_layers=1,
        bidirectional=False, dropout=0.0,
    )
    config = {
        "device": "cpu",
        "pos_weight": 5.0,
        "lr": 0.001,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "mixed_precision": False,
        "patience": 3,
        "checkpoint_dir": "/tmp/test_sepsis_checkpoints",
        "log_dir": "/tmp/test_sepsis_logs",
    }
    if config_overrides:
        config.update(config_overrides)

    return SepsisTrainer(model, loader, loader, config)


class TestSepsisTrainer:
    def test_init(self):
        trainer = _make_trainer()
        assert trainer.device == torch.device("cpu")
        assert trainer.pos_weight == 5.0

    def test_auto_pos_weight(self):
        trainer = _make_trainer(config_overrides={"pos_weight": "auto"})
        # 10 patients * 10 hours = 100 samples, 2 sepsis * 2 hours = 4 positive
        # pos_weight = 96/4 = 24.0
        assert trainer.pos_weight == pytest.approx(24.0, rel=0.1)

    def test_train_epoch(self):
        trainer = _make_trainer()
        loss = trainer.train_epoch()
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert loss > 0

    def test_validate(self):
        trainer = _make_trainer()
        val_loss, preds, labels, pids, hours = trainer.validate()
        assert isinstance(val_loss, float)
        assert not np.isnan(val_loss)
        assert len(preds) == len(labels) == len(pids) == len(hours)
        # Predictions should be probabilities in [0, 1]
        assert all(0 <= p <= 1 for p in preds)

    def test_find_optimal_threshold(self):
        trainer = _make_trainer()
        _, preds, labels, pids, hours = trainer.validate()
        threshold, utility = trainer.find_optimal_threshold(preds, labels, pids, hours)
        assert 0.1 <= threshold <= 0.9
        assert isinstance(utility, float)

    def test_train_loop(self):
        trainer = _make_trainer()
        history = trainer.train(epochs=3)
        assert len(history["train_loss"]) == 3
        assert len(history["val_utility"]) == 3
        assert all(isinstance(v, float) for v in history["train_loss"])

    def test_early_stopping(self):
        # With patience=1 and random data, early stopping should trigger
        trainer = _make_trainer(config_overrides={"patience": 1})
        history = trainer.train(epochs=20)
        # Should stop well before 20 epochs
        assert len(history["train_loss"]) < 20

    def test_checkpoint_saved(self):
        import os
        import shutil
        ckpt_dir = "/tmp/test_sepsis_ckpt_save"
        trainer = _make_trainer(config_overrides={"checkpoint_dir": ckpt_dir})
        trainer.train(epochs=2)
        assert os.path.exists(os.path.join(ckpt_dir, "gru", "best.pt"))
        # Clean up
        shutil.rmtree(ckpt_dir)
