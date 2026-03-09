"""Preprocessed data caching to avoid reprocessing ~40k patients every run."""

import hashlib
import os
import pickle
import logging

logger = logging.getLogger("sepsis")


def get_cache_path(config: dict, cache_dir: str = "data/processed") -> str:
    """Generate cache filepath based on config hash.

    Hashes seq_length, val_split, max_patients, and input_size to produce
    a deterministic filename that changes when relevant settings change.
    """
    key_parts = {
        "seq_length": config.get("data", {}).get("seq_length"),
        "val_split": config.get("data", {}).get("val_split"),
        "max_patients": config.get("data", {}).get("max_patients"),
        "input_size": config.get("model", {}).get("input_size"),
        "seed": config.get("seed"),
    }
    h = hashlib.sha256(str(sorted(key_parts.items())).encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f"preprocessed_cache_{h}.pkl")


def is_cache_valid(cache_path: str, raw_data_dir: str) -> bool:
    """Check if cache exists and is newer than the raw data directory.

    Returns False if cache doesn't exist or if any raw data subdirectory
    has been modified after the cache was created.
    """
    if not os.path.exists(cache_path):
        return False

    cache_mtime = os.path.getmtime(cache_path)

    for subdir in ["training_setA", "training_setB"]:
        path = os.path.join(raw_data_dir, subdir)
        if os.path.isdir(path):
            if os.path.getmtime(path) > cache_mtime:
                return False

    return True


def save_preprocessed(
    train_processed: dict,
    val_processed: dict,
    train_ids: list,
    val_ids: list,
    train_stats: dict,
    filepath: str,
) -> None:
    """Save preprocessed data to a pickle file.

    Args:
        train_processed: Dict of patient_id -> preprocessed DataFrame (train).
        val_processed: Dict of patient_id -> preprocessed DataFrame (val).
        train_ids: List of training patient IDs.
        val_ids: List of validation patient IDs.
        train_stats: Population-level imputation/normalization stats.
        filepath: Output path for the pickle file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        "train_processed": train_processed,
        "val_processed": val_processed,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_stats": train_stats,
    }
    with open(filepath, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logger.info("Saved preprocessed cache to %s (%.1f MB)", filepath, size_mb)


def load_preprocessed(filepath: str) -> dict:
    """Load preprocessed data from a pickle file.

    Returns:
        Dict with keys: train_processed, val_processed, train_ids, val_ids, train_stats.
    """
    with open(filepath, "rb") as f:
        payload = pickle.load(f)
    logger.info("Loaded preprocessed cache from %s", filepath)
    return payload
