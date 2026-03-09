"""Tests for data loading and statistics utilities."""

from src.data.preprocessing import (
    ALL_FEATURES,
    CLINICAL_FEATURES,
    extract_patient_id,
    load_all_patients,
    compute_statistics,
    print_statistics,
)

DATA_DIR = "data/raw/training_setA"


def test_extract_patient_id():
    assert extract_patient_id("p000001.psv") == "p000001"
    assert extract_patient_id("/some/path/p123456.psv") == "p123456"


def test_load_subset():
    patients = load_all_patients(DATA_DIR, max_patients=10)
    assert len(patients) == 10

    # Check that patient IDs are strings starting with 'p'
    for pid in patients:
        assert pid.startswith("p")

    # Check DataFrame shape and columns for each patient
    for pid, df in patients.items():
        assert list(df.columns) == ALL_FEATURES, f"Column mismatch for {pid}"
        assert len(df) > 0, f"Empty DataFrame for {pid}"
        assert df.shape[1] == 41, f"Expected 41 columns, got {df.shape[1]} for {pid}"


def test_compute_statistics():
    patients = load_all_patients(DATA_DIR, max_patients=10)
    stats = compute_statistics(patients)

    assert stats["n_patients"] == 10
    assert stats["total_records"] > 0
    assert 0.0 <= stats["sepsis_prevalence"] <= 1.0
    assert stats["seq_length"]["min"] <= stats["seq_length"]["max"]
    assert stats["seq_length"]["min"] >= 1

    # Missingness rates should be between 0 and 1
    for feat, rate in stats["missingness"].items():
        assert 0.0 <= rate <= 1.0, f"Invalid missingness for {feat}: {rate}"
        assert feat in CLINICAL_FEATURES

    print_statistics(stats)
