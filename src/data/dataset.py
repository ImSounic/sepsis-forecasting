"""PyTorch Dataset and DataLoader for sepsis prediction."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SepsisDataset(Dataset):
    """Sliding-window dataset for hourly sepsis prediction.

    For each patient hour, creates a window of the past seq_length hours.
    Early hours (fewer than seq_length available) are left-padded with zeros.

    Uses per-patient numpy arrays with fast __getitem__ indexing.
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

        # Convert DataFrames to numpy once; pre-pad each patient for O(1) getitem
        self.patient_data = []  # list of (features_padded, labels, pid)
        self.samples = []       # list of (patient_idx, hour)

        for pid, df in patients.items():
            feat = df[self.feature_columns].values.astype(np.float32)
            labels = df[label_column].values.astype(np.float32)

            # Pre-pad with seq_length-1 zeros so window extraction is a simple slice
            pad = np.zeros((seq_length - 1, self.num_features), dtype=np.float32)
            feat_padded = np.concatenate([pad, feat], axis=0)

            patient_idx = len(self.patient_data)
            self.patient_data.append((feat_padded, labels, pid))

            for hour in range(len(df)):
                self.samples.append((patient_idx, hour))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_idx, hour = self.samples[idx]
        feat_padded, labels, pid = self.patient_data[patient_idx]

        # Simple slice — no branching, no concatenation
        window = feat_padded[hour:hour + self.seq_length]
        label = labels[hour]

        # Mask: 1 where real data, 0 where padding
        actual_len = min(hour + 1, self.seq_length)
        mask = np.ones(self.seq_length, dtype=np.float32)
        if actual_len < self.seq_length:
            mask[:self.seq_length - actual_len] = 0.0

        return (
            torch.from_numpy(window),
            torch.tensor(label, dtype=torch.float32),
            torch.from_numpy(mask),
            pid,
            hour,
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
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
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
