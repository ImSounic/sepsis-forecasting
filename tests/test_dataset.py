"""Tests for PyTorch Dataset and DataLoader."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SepsisDataset, create_data_loaders, get_train_val_split


def make_test_patient(n_hours, n_features=4, sepsis_onset=None):
    """Create a preprocessed patient with sequential feature values for easy verification.

    Row i has features [i*n_features, i*n_features+1, ...].
    """
    features = np.arange(n_hours * n_features, dtype=np.float32).reshape(n_hours, n_features)
    labels = np.zeros(n_hours, dtype=np.float32)
    if sepsis_onset is not None:
        labels[sepsis_onset:] = 1.0

    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=cols)
    df["SepsisLabel"] = labels
    return df


class TestSepsisDataset:
    def test_window_count_single_patient(self):
        df = make_test_patient(10)
        dataset = SepsisDataset({"p1": df}, seq_length=3)
        # One window per hour
        assert len(dataset) == 10

    def test_window_count_multiple_patients(self):
        patients = {"p1": make_test_patient(10), "p2": make_test_patient(5)}
        dataset = SepsisDataset(patients, seq_length=3)
        assert len(dataset) == 15

    def test_full_window_no_padding(self):
        df = make_test_patient(5, n_features=4)
        dataset = SepsisDataset({"p1": df}, seq_length=3)

        # Hour 3: window is rows [1, 2, 3], full sequence, no padding
        features, label, mask, pid, hour = dataset[3]
        expected = np.array([[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], dtype=np.float32)
        np.testing.assert_array_equal(features.numpy(), expected)
        assert mask.tolist() == [1.0, 1.0, 1.0]
        assert label.item() == 0.0
        assert pid == "p1"
        assert hour == 3

    def test_fully_padded_first_hour(self):
        df = make_test_patient(5, n_features=4)
        dataset = SepsisDataset({"p1": df}, seq_length=3)

        # Hour 0: only row 0 available, 2 padding rows on left
        features, label, mask, pid, hour = dataset[0]
        expected = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 2, 3]], dtype=np.float32)
        np.testing.assert_array_equal(features.numpy(), expected)
        assert mask.tolist() == [0.0, 0.0, 1.0]

    def test_partial_padding(self):
        df = make_test_patient(5, n_features=4)
        dataset = SepsisDataset({"p1": df}, seq_length=3)

        # Hour 1: rows [0, 1] available, 1 padding row
        features, _, mask, _, _ = dataset[1]
        expected = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.float32)
        np.testing.assert_array_equal(features.numpy(), expected)
        assert mask.tolist() == [0.0, 1.0, 1.0]

    def test_short_patient_all_padded(self):
        df = make_test_patient(2, n_features=2)
        dataset = SepsisDataset({"p1": df}, seq_length=5)
        assert len(dataset) == 2

        # Hour 1: 2 real rows + 3 padding
        features, _, mask, _, _ = dataset[1]
        assert features.shape == (5, 2)
        assert mask.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]

    def test_sepsis_labels(self):
        df = make_test_patient(5, sepsis_onset=3)
        dataset = SepsisDataset({"p1": df}, seq_length=2)

        _, label2, _, _, _ = dataset[2]
        _, label3, _, _, _ = dataset[3]
        _, label4, _, _, _ = dataset[4]
        assert label2.item() == 0.0
        assert label3.item() == 1.0
        assert label4.item() == 1.0

    def test_tensor_shapes_120_features(self):
        df = make_test_patient(10, n_features=120)
        dataset = SepsisDataset({"p1": df}, seq_length=24)
        features, label, mask, pid, hour = dataset[0]
        assert features.shape == (24, 120)
        assert label.shape == ()
        assert mask.shape == (24,)

    def test_auto_detects_feature_columns(self):
        df = make_test_patient(5, n_features=4)
        dataset = SepsisDataset({"p1": df}, seq_length=2)
        assert dataset.feature_columns == ["f0", "f1", "f2", "f3"]
        assert dataset.num_features == 4


class TestDataLoaderIntegration:
    def test_batch_shapes(self):
        patients = {f"p{i}": make_test_patient(10) for i in range(5)}
        dataset = SepsisDataset(patients, seq_length=3)
        loader = DataLoader(dataset, batch_size=8, num_workers=0)

        features, labels, masks, pids, hours = next(iter(loader))
        assert features.shape == (8, 3, 4)
        assert labels.shape == (8,)
        assert masks.shape == (8, 3)
        assert len(pids) == 8

    def test_last_batch_smaller(self):
        # 3 patients * 5 hours = 15 samples, batch_size=10 -> batches of 10 and 5
        patients = {f"p{i}": make_test_patient(5) for i in range(3)}
        dataset = SepsisDataset(patients, seq_length=2)
        loader = DataLoader(dataset, batch_size=10, num_workers=0)
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0][0].shape[0] == 10
        assert batches[1][0].shape[0] == 5

    def test_create_data_loaders(self):
        train = {f"p{i}": make_test_patient(10) for i in range(20)}
        val = {f"v{i}": make_test_patient(10) for i in range(5)}
        train_loader, val_loader = create_data_loaders(
            train, val, batch_size=16, seq_length=3, num_workers=0,
        )
        assert len(train_loader.dataset) == 200
        assert len(val_loader.dataset) == 50


class TestTrainValSplit:
    def test_split_sizes(self):
        ids = [f"p{i:04d}" for i in range(100)]
        train, val = get_train_val_split(ids, val_ratio=0.2, seed=42)
        assert len(train) == 80
        assert len(val) == 20

    def test_no_overlap(self):
        ids = [f"p{i:04d}" for i in range(100)]
        train, val = get_train_val_split(ids, val_ratio=0.2, seed=42)
        assert set(train).isdisjoint(set(val))
        assert set(train) | set(val) == set(ids)

    def test_reproducible(self):
        ids = [f"p{i:04d}" for i in range(100)]
        train1, val1 = get_train_val_split(ids, seed=42)
        train2, val2 = get_train_val_split(ids, seed=42)
        assert train1 == train2
        assert val1 == val2

    def test_different_seeds_differ(self):
        ids = [f"p{i:04d}" for i in range(100)]
        train1, _ = get_train_val_split(ids, seed=42)
        train2, _ = get_train_val_split(ids, seed=123)
        assert train1 != train2
