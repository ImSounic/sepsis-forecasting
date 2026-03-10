"""Main entry point for LightGBM baseline training.

Usage:
    python -m src.training.train_baseline --config configs/default.yaml
    python -m src.training.train_baseline --config configs/default.yaml --improved
    python -m src.training.train_baseline --config configs/default.yaml --max_patients 50
"""

import argparse
import gc
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from src.data.cache import get_cache_path, is_cache_valid, load_preprocessed, save_preprocessed
from src.data.dataset import get_train_val_split
from src.data.feature_engineering import INPUT_FEATURES
from src.data.preprocessing import TARGET, load_all_patients
from src.models.lightgbm_baseline import (
    get_baseline_feature_names as get_baseline_names,
    train_and_evaluate_baseline as train_baseline,
)
from src.models.lightgbm_model import (
    get_baseline_feature_names as get_improved_names,
    train_and_evaluate_baseline as train_improved,
)
from src.training.train import compute_train_stats, load_config, preprocess_patients, set_seed, setup_logging


def main(config_path: str, max_patients: int = None, no_cache: bool = False,
         improved: bool = False):
    config = load_config(config_path)
    logger = setup_logging(config["output"]["log_dir"])
    seed = config.get("seed", 42)
    set_seed(seed)

    mode = "improved" if improved else "default"
    logger.info("LightGBM mode: %s", mode)

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

        del patients, train_raw, val_raw
        gc.collect()

        # Save to cache
        save_preprocessed(train_processed, val_processed, train_ids, val_ids, train_stats, cache_path)

    # Get the 120 feature columns (everything except SepsisLabel)
    sample_df = next(iter(train_processed.values()))
    feature_columns = [c for c in sample_df.columns if c != TARGET]

    # Train and evaluate
    logger.info("Training LightGBM (improved=%s)...", improved)
    if improved:
        model, threshold, utility, importance = train_improved(
            train_processed, val_processed, feature_columns, improved=True,
        )
        feature_names = get_improved_names(feature_columns, enhanced=True)
    else:
        model, threshold, utility, importance = train_baseline(
            train_processed, val_processed, feature_columns,
        )
        feature_names = get_baseline_names(feature_columns)

    logger.info("Utility: %.4f | Threshold: %.2f", utility, threshold)

    # Save model
    model_dir = config["output"]["model_dir"]
    lgb_dir = os.path.join(model_dir, "lightgbm")
    os.makedirs(lgb_dir, exist_ok=True)
    filename = "lightgbm_improved.pkl" if improved else "lightgbm_baseline.pkl"
    model_path = os.path.join(lgb_dir, filename)
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    # Log top features
    logger.info("Top 20 features by importance:")
    for _, row in importance.head(20).iterrows():
        logger.info("  %s: %.1f", row["feature"], row["importance"])

    n_base = len(feature_columns)
    n_total = len(feature_names)
    logger.info("Features: %d base + %d engineered = %d total", n_base, n_total - n_base, n_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM baseline for sepsis prediction")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--max_patients", type=int, default=None,
        help="Limit patients loaded (per dataset directory) for quick debugging",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force reprocessing even if a valid cache exists",
    )
    parser.add_argument(
        "--improved", action="store_true",
        help="Use improved hyperparameters and enhanced features",
    )
    args = parser.parse_args()
    main(args.config, max_patients=args.max_patients, no_cache=args.no_cache,
         improved=args.improved)
