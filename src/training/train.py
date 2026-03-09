"""Main training script for sepsis prediction.

Usage:
    python -m src.training.train --config configs/default.yaml
    python -m src.training.train --config configs/default.yaml --max_patients 100
"""

import argparse
import gc
import logging
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
import yaml

from src.data.cache import get_cache_path, is_cache_valid, load_preprocessed, save_preprocessed
from src.data.dataset import create_data_loaders, get_train_val_split
from src.data.feature_engineering import (
    INPUT_FEATURES,
    forward_fill_patient,
    preprocess_patient,
)
from src.data.preprocessing import TARGET, load_all_patients
from src.models.gru import GRUWithAttention, count_parameters
from src.training.trainer import SepsisTrainer


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging to console and file."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("sepsis")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)

    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    return logger


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _compute_norm_stats_for_columns(data: pd.DataFrame, columns: list[str]) -> dict:
    """Compute winsorized z-score normalization stats for given columns."""
    norm_stats = {}
    for col in columns:
        if col not in data.columns:
            continue
        p1 = float(data[col].quantile(0.01))
        p99 = float(data[col].quantile(0.99))
        if pd.isna(p1) or pd.isna(p99):
            p1, p99 = 0.0, 0.0
        clipped = data[col].clip(p1, p99)
        mean_val = float(clipped.mean())
        std_val = float(clipped.std())
        if pd.isna(mean_val):
            mean_val = 0.0
        if std_val == 0 or pd.isna(std_val):
            std_val = 1.0
        norm_stats[col] = {"mean": mean_val, "std": std_val, "p1": p1, "p99": p99}
    return norm_stats


def compute_train_stats(patients: dict[str, pd.DataFrame]) -> dict:
    """Compute population-level imputation and normalization stats from training patients.

    Steps:
      1. Forward-fill each patient, combine for imputation stats.
      2. Compute normalization stats for 40 features.
    """
    logger = logging.getLogger("sepsis")

    # Step 1: Forward-fill all patients, combine for imputation stats
    filled = pd.concat(
        [forward_fill_patient(df) for df in patients.values()],
        ignore_index=True,
    )

    impute_stats = {}
    for col in INPUT_FEATURES:
        val = filled[col].median()
        impute_stats[col] = 0.0 if pd.isna(val) else float(val)

    # Apply imputation and compute normalization stats for original features
    for col in INPUT_FEATURES:
        filled[col] = filled[col].fillna(impute_stats[col])

    norm_stats = _compute_norm_stats_for_columns(filled, INPUT_FEATURES)
    del filled
    gc.collect()

    return {"impute": impute_stats, "normalize": norm_stats}


def preprocess_patients(
    raw_patients: dict[str, pd.DataFrame], train_stats: dict, batch_size: int = 1000
) -> dict[str, pd.DataFrame]:
    """Preprocess all patients and attach labels.

    Processes in batches to limit peak memory usage, running gc.collect()
    between batches.

    Returns dict of DataFrames with 120 feature columns + SepsisLabel.
    """
    logger = logging.getLogger("sepsis")
    pids = list(raw_patients.keys())
    n_batches = (len(pids) + batch_size - 1) // batch_size
    processed = {}

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(pids))
        batch_pids = pids[start:end]
        logger.info("Preprocessing batch %d/%d (%d patients)...", batch_idx + 1, n_batches, len(batch_pids))

        for pid in batch_pids:
            df = raw_patients[pid]
            features, _ = preprocess_patient(df, train_stats=train_stats)
            label_col = pd.DataFrame({TARGET: df[TARGET].values}, index=features.index)
            processed[pid] = pd.concat([features, label_col], axis=1)

        gc.collect()

    return processed


def main(config_path: str, max_patients: int = None, no_cache: bool = False):
    config = load_config(config_path)
    logger = setup_logging(config["output"]["log_dir"])
    seed = config.get("seed", 42)
    set_seed(seed)

    # CLI override for max_patients
    if max_patients is not None:
        config["data"]["max_patients"] = max_patients

    raw_dir = config["data"]["raw_dir"]
    cache_path = get_cache_path(config)

    # Try loading from cache
    if not no_cache and is_cache_valid(cache_path, raw_dir):
        logger.info("Valid cache found, loading preprocessed data from %s", cache_path)
        cached = load_preprocessed(cache_path)
        train_processed = cached["train_processed"]
        val_processed = cached["val_processed"]
        train_ids = cached["train_ids"]
        val_ids = cached["val_ids"]
        train_stats = cached["train_stats"]
        logger.info("Loaded %d train, %d val patients from cache", len(train_processed), len(val_processed))
    else:
        if no_cache:
            logger.info("Cache disabled via --no-cache, processing fresh")
        else:
            logger.info("No valid cache found, processing from raw data")

        max_p = config["data"].get("max_patients")

        # Load raw data
        logger.info("Loading raw patient data from %s ...", raw_dir)
        patients = {}
        for subdir in ["training_setA", "training_setB"]:
            path = os.path.join(raw_dir, subdir)
            if os.path.isdir(path):
                sub = load_all_patients(path, max_patients=max_p)
                patients.update(sub)
                logger.info("  %s: %d patients", subdir, len(sub))

        if not patients:
            logger.error("No patient data found in %s", raw_dir)
            return

        logger.info("Total patients: %d", len(patients))

        # Train/val split
        train_ids, val_ids = get_train_val_split(
            list(patients.keys()),
            val_ratio=config["data"]["val_split"],
            seed=seed,
        )
        train_raw = {pid: patients[pid] for pid in train_ids}
        val_raw = {pid: patients[pid] for pid in val_ids}
        logger.info("Split: %d train, %d val", len(train_raw), len(val_raw))

        # Compute population stats from training data
        logger.info("Computing training statistics...")
        train_stats = compute_train_stats(train_raw)

        # Preprocess all patients
        logger.info("Preprocessing %d training patients...", len(train_raw))
        train_processed = preprocess_patients(train_raw, train_stats)
        logger.info("Preprocessing %d validation patients...", len(val_raw))
        val_processed = preprocess_patients(val_raw, train_stats)

        # Free raw data
        del patients, train_raw, val_raw
        gc.collect()

        # Save to cache
        save_preprocessed(train_processed, val_processed, train_ids, val_ids, train_stats, cache_path)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_processed,
        val_processed,
        batch_size=config["training"]["batch_size"],
        seq_length=config["data"]["seq_length"],
        num_workers=config["training"].get("num_workers", 4),
    )
    logger.info(
        "DataLoaders: %d train samples, %d val samples",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # Build model
    model = GRUWithAttention(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        bidirectional=config["model"]["bidirectional"],
        attention_size=config["model"].get("attention_size", 64),
    )
    total, trainable = count_parameters(model)
    logger.info("Model: %s params (%s trainable)", f"{total:,}", f"{trainable:,}")

    # Initialize trainer
    trainer_config = {
        "device": config.get("device", "auto"),
        "pos_weight": config["training"].get("pos_weight", "auto"),
        "lr": config["training"]["lr"],
        "weight_decay": config["training"]["weight_decay"],
        "max_grad_norm": config["training"]["gradient_clip"],
        "mixed_precision": config["training"].get("mixed_precision", True),
        "patience": config["training"]["patience"],
        "scheduler_patience": config["training"].get("scheduler_patience", 5),
        "scheduler_factor": config["training"].get("scheduler_factor", 0.5),
        "min_lr": config["training"].get("min_lr", 1e-5),
        "checkpoint_dir": config["output"]["model_dir"],
        "log_dir": config["output"]["log_dir"],
    }
    trainer = SepsisTrainer(model, train_loader, val_loader, trainer_config)
    logger.info("Positive class weight: %.2f", trainer.pos_weight)

    # Train
    logger.info("Starting training for %d epochs...", config["training"]["epochs"])
    history = trainer.train(config["training"]["epochs"])

    # Final report
    if history["val_utility"]:
        best_idx = int(np.argmax(history["val_utility"]))
        logger.info(
            "Done. Best epoch %d: utility=%.4f, threshold=%.2f, val_loss=%.4f",
            best_idx + 1,
            history["val_utility"][best_idx],
            history["threshold"][best_idx],
            history["val_loss"][best_idx],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sepsis prediction model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--max_patients", type=int, default=None,
        help="Limit patients loaded (per dataset directory) for quick debugging",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force reprocessing even if a valid cache exists",
    )
    args = parser.parse_args()
    main(args.config, max_patients=args.max_patients, no_cache=args.no_cache)
