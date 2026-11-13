import pytest
import pandas as pd
import numpy as np
from src.data import generate_hillstrom_synthetic, preprocess_data, HILLSTROM_FEATURE_COLS


def test_generate_hillstrom_synthetic_schema():
    df = generate_hillstrom_synthetic(n_samples=500, random_state=42)
    assert len(df) == 500
    expected_cols = [
        "recency", "history", "mens", "womens", "zip_code",
        "newbie", "channel", "segment", "treatment", "visit", "conversion", "spend"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Treatment should be binary 0 or 1
    assert set(df["treatment"].unique()).issubset({0, 1})
    # Conversion should be binary 0 or 1
    assert set(df["conversion"].unique()).issubset({0, 1})
    # Spend should be non-negative
    assert (df["spend"] >= 0).all()


def test_preprocess_data_encoding():
    df = generate_hillstrom_synthetic(n_samples=200, random_state=42)
    X, treatment, y_conv, y_spend = preprocess_data(df)

    assert X.shape[0] == 200
    assert len(treatment) == 200
    assert len(y_conv) == 200
    assert len(y_spend) == 200

    # Ensure all required features are encoded
    for feat in HILLSTROM_FEATURE_COLS:
        assert feat in X.columns
        assert not X[feat].isna().any()
