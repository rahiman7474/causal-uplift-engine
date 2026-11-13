import pytest
import pandas as pd
import numpy as np
from src.data import generate_hillstrom_synthetic, preprocess_data
from src.balance import calculate_smd, check_overlap_and_positivity


def test_calculate_smd_balanced():
    # Randomized dataset should have all SMD < 0.15
    df = generate_hillstrom_synthetic(n_samples=2000, random_state=42)
    X, treatment, _, _ = preprocess_data(df)
    smd_df = calculate_smd(X, treatment)

    assert "smd" in smd_df.columns
    assert "status" in smd_df.columns
    assert len(smd_df) == X.shape[1]

    # In an RCT, max SMD should be small (< 0.15)
    assert smd_df["smd"].max() < 0.15


def test_positivity_overlap():
    df = generate_hillstrom_synthetic(n_samples=1000, random_state=42)
    X, treatment, _, _ = preprocess_data(df)
    overlap = check_overlap_and_positivity(X, treatment)

    assert "positivity_satisfied" in overlap
    assert "overlap_window" in overlap
    assert overlap["overlap_window"][0] > 0.0
    assert overlap["overlap_window"][1] < 1.0
    assert overlap["positivity_satisfied"] is True
