import pytest
import pandas as pd
import numpy as np
from src.data import generate_hillstrom_synthetic, preprocess_data
from src.models import TLearner
from src.refutation import placebo_treatment_refutation, random_common_cause_refutation


def test_placebo_refutation():
    df = generate_hillstrom_synthetic(n_samples=500, random_state=42)
    X, treatment, y_conv, _ = preprocess_data(df)

    res = placebo_treatment_refutation(TLearner, X, treatment, y_conv, n_simulations=2, random_state=42)
    assert "test_name" in res
    assert "mean_placebo_ate" in res
    assert "refutation_passed" in res
    # Mean placebo effect should be close to zero
    assert abs(res["mean_placebo_ate"]) < 0.05


def test_random_common_cause_refutation():
    df = generate_hillstrom_synthetic(n_samples=500, random_state=42)
    X, treatment, y_conv, _ = preprocess_data(df)

    res = random_common_cause_refutation(TLearner, X, treatment, y_conv, random_state=42)
    assert "test_name" in res
    assert "drift_pct" in res
    assert "refutation_passed" in res
