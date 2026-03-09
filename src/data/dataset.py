"""PyTorch Dataset and DataLoader for sepsis prediction."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SepsisDataset(Dataset):
    """Sliding-window dataset for hourly sepsis prediction.

    For each patient hour, creates a window of the past seq_length hours.
    Early hours (fewer than seq_length available) are left-padded with zeros.
    """

    def __init__(
        self,
        patients: dict[str, pd.DataFrame],
        seq_length: int = 24,
        feature_columns: list[str] = None,
        label_column: str = "SepsisLabel",
    ):
        self.seq_length = seq_length
        self.label_column = label_column

        # Auto-detect feature columns if not provided
        if feature_columns is None:
            sample_df = next(iter(patients.values()))
            feature_columns = [c for c in sample_df.columns if c != label_column]
        self.feature_columns = list(feature_columns)
        self.num_features = len(self.feature_columns)

        # Convert DataFrames to numpy for fast indexing in __getitem__
        self.patient_features = {}
        self.patient_labels = {}
        self.samples = []

        for pid, df in patients.items():
            self.patient_features[pid] = df[self.feature_columns].values.astype(np.float32)
            self.patient_labels[pid] = df[label_column].values.astype(np.float32)
            for hour in range(len(df)):
                self.samples.append((pid, hour))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, hour = self.samples[idx]
        features = self.patient_features[pid]
        labels = self.patient_labels[pid]

        # Window: past seq_length hours ending at current hour (inclusive)
        start = max(0, hour + 1 - self.seq_length)
        end = hour + 1
        window = features[start:end]
        label = labels[hour]

        # Left-pad with zeros if window is shorter than seq_length
        actual_len = window.shape[0]
        if actual_len < self.seq_length:
            pad_len = self.seq_length - actual_len
            padding = np.zeros((pad_len, self.num_features), dtype=np.float32)
            window = np.concatenate([padding, window], axis=0)
            mask = np.concatenate([
                np.zeros(pad_len, dtype=np.float32),
                np.ones(actual_len, dtype=np.float32),
            ])
        else:
            mask = np.ones(self.seq_length, dtype=np.float32)

        return (
            torch.from_numpy(window),  # (seq_length, num_features)
            torch.tensor(label),       # scalar
            torch.from_numpy(mask),    # (seq_length,)
            pid,                       # str
            hour,                      # int
        )


def create_data_loaders(
    train_patients: dict[str, pd.DataFrame],
    val_patients: dict[str, pd.DataFrame],
    batch_size: int = 64,
    seq_length: int = 24,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from preprocessed patient dicts.

    Expects DataFrames with feature columns + 'SepsisLabel'.
    """
    train_dataset = SepsisDataset(train_patients, seq_length=seq_length)
    val_dataset = SepsisDataset(val_patients, seq_length=seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_train_val_split(
    patient_ids, val_ratio: float = 0.15, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split patient IDs into train and validation sets.

    Returns:
        (train_ids, val_ids)
    """
    rng = np.random.RandomState(seed)
    ids = list(patient_ids)
    rng.shuffle(ids)
    split_idx = int(len(ids) * (1 - val_ratio))
    return ids[:split_idx], ids[split_idx:]
