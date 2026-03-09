"""Data loading and statistics for PhysioNet 2019 Challenge PSV files."""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# PhysioNet 2019 feature groups
VITAL_SIGNS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LAB_VALUES = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
TARGET = "SepsisLabel"

CLINICAL_FEATURES = VITAL_SIGNS + LAB_VALUES
ALL_FEATURES = CLINICAL_FEATURES + DEMOGRAPHICS + [TARGET]


def extract_patient_id(filename: str) -> str:
    """Extract patient ID from PSV filename. e.g. 'p000001.psv' -> 'p000001'."""
    return Path(filename).stem


def load_psv(filepath: str) -> pd.DataFrame:
    """Load a single PSV (pipe-separated) file into a DataFrame."""
    return pd.read_csv(filepath, sep="|")


def load_all_patients(data_dir: str, max_patients: int = None) -> dict[str, pd.DataFrame]:
    """Load all PSV files from a directory into a dict keyed by patient ID.

    Args:
        data_dir: Path to directory containing PSV files.
        max_patients: If set, only load this many patients (sorted by filename).

    Returns:
        Dictionary mapping patient_id -> DataFrame.
    """
    data_dir = Path(data_dir)
    psv_files = sorted(data_dir.glob("*.psv"))

    if max_patients is not None:
        psv_files = psv_files[:max_patients]

    patients = {}
    for filepath in psv_files:
        pid = extract_patient_id(filepath.name)
        patients[pid] = load_psv(filepath)

    return patients


def compute_statistics(patients: dict[str, pd.DataFrame]) -> dict:
    """Compute dataset statistics from loaded patient data.

    Args:
        patients: Dictionary mapping patient_id -> DataFrame.

    Returns:
        Dictionary with dataset statistics.
    """
    n_patients = len(patients)
    seq_lengths = np.array([len(df) for df in patients.values()])
    total_records = int(seq_lengths.sum())

    # Missingness rate per clinical feature (exclude demographics and target)
    combined = pd.concat(patients.values(), ignore_index=True)
    missingness = combined[CLINICAL_FEATURES].isna().mean().to_dict()

    # Sepsis prevalence: fraction of patients with at least one SepsisLabel=1
    sepsis_count = sum(
        1 for df in patients.values() if df[TARGET].max() == 1
    )
    sepsis_prevalence = sepsis_count / n_patients if n_patients > 0 else 0.0

    return {
        "n_patients": n_patients,
        "total_records": total_records,
        "sepsis_patients": sepsis_count,
        "sepsis_prevalence": sepsis_prevalence,
        "seq_length": {
            "min": int(seq_lengths.min()),
            "max": int(seq_lengths.max()),
            "mean": float(seq_lengths.mean()),
            "median": float(np.median(seq_lengths)),
        },
        "missingness": missingness,
    }


def print_statistics(stats: dict) -> None:
    """Print dataset statistics in a readable format."""
    print(f"Patients: {stats['n_patients']}")
    print(f"Total hourly records: {stats['total_records']}")
    print(f"Sepsis patients: {stats['sepsis_patients']} ({stats['sepsis_prevalence']:.2%})")
    print(
        f"Sequence length: min={stats['seq_length']['min']}, "
        f"max={stats['seq_length']['max']}, "
        f"mean={stats['seq_length']['mean']:.1f}, "
        f"median={stats['seq_length']['median']:.0f}"
    )
    print("\nMissingness by feature:")
    for feat, rate in sorted(stats["missingness"].items(), key=lambda x: -x[1]):
        print(f"  {feat:>20s}: {rate:.1%}")
