import pytest
import numpy as np
import pandas as pd
from src.evaluation import compute_qini_curve, compute_decile_table


def test_qini_curve_structure():
    np.random.seed(42)
    n = 1000
    y = np.random.binomial(1, 0.05, size=n)
    t = np.random.binomial(1, 0.5, size=n)
    preds = np.random.normal(0.01, 0.02, size=n)

    qini = compute_qini_curve(y, t, preds, n_bins=20)
    assert "fractions" in qini
    assert "qini_values" in qini
    assert "random_values" in qini
    assert "qini_score" in qini
    assert len(qini["fractions"]) == 20
    assert len(qini["qini_values"]) == 20


def test_decile_table_monotonicity():
    np.random.seed(42)
    n = 1000
    y = np.random.binomial(1, 0.05, size=n)
    t = np.random.binomial(1, 0.5, size=n)
    preds = np.linspace(0.05, -0.01, n)

    dec_df = compute_decile_table(y, t, preds)
    assert len(dec_df) == 10
    assert "empirical_uplift" in dec_df.columns
    assert "mean_pred_uplift" in dec_df.columns
    # First decile should have higher mean predicted uplift than 10th
    assert dec_df.loc[0, "mean_pred_uplift"] > dec_df.loc[9, "mean_pred_uplift"]
