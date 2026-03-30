"""PyTorch Dataset and DataLoader for sepsis prediction."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SepsisDataset(Dataset):
    """Sliding-window dataset for hourly sepsis prediction.

    For each patient hour, creates a window of the past seq_length hours.
    Early hours (fewer than seq_length available) are left-padded with zeros.

    All windows are pre-computed in __init__ for fast __getitem__ access.
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

        # Count total samples first
        total_samples = sum(len(df) for df in patients.values())

        # Pre-allocate tensors for all windows
        self.all_features = torch.zeros(
            (total_samples, seq_length, self.num_features), dtype=torch.float32
        )
        self.all_labels = torch.zeros(total_samples, dtype=torch.float32)
        self.all_masks = torch.zeros(
            (total_samples, seq_length), dtype=torch.float32
        )
        self.sample_pids = []
        self.sample_hours = []

        idx = 0
        for pid, df in patients.items():
            feat = df[self.feature_columns].values.astype(np.float32)
            labels = df[label_column].values.astype(np.float32)

            for hour in range(len(df)):
                start = max(0, hour + 1 - seq_length)
                end = hour + 1
                window = feat[start:end]
                actual_len = window.shape[0]
                pad_len = seq_length - actual_len

                # Write directly into pre-allocated tensor
                self.all_features[idx, pad_len:] = torch.from_numpy(window)
                self.all_labels[idx] = labels[hour]
                self.all_masks[idx, pad_len:] = 1.0

                self.sample_pids.append(pid)
                self.sample_hours.append(hour)
                idx += 1

    def __len__(self):
        return len(self.sample_pids)

    def __getitem__(self, idx):
        return (
            self.all_features[idx],    # (seq_length, num_features)
            self.all_labels[idx],      # scalar
            self.all_masks[idx],       # (seq_length,)
            self.sample_pids[idx],     # str
            self.sample_hours[idx],    # int
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
