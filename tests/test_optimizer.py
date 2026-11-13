import pytest
import numpy as np
from src.optimizer import optimize_targeting_policy, compute_roi_curve, classify_customer_quadrants


def test_optimize_targeting_policy():
    np.random.seed(42)
    uplifts = np.array([0.05, 0.03, 0.01, 0.002, -0.01, -0.05] * 100)
    res = optimize_targeting_policy(uplifts, unit_cost=0.10, conversion_value=30.0)

    assert "targeted_customers" in res
    assert "net_profit" in res
    assert "blanket_profit" in res
    assert "profit_lift_vs_blanket" in res
    # Targeted fraction should be strictly less than 1.0 (excluding negatives)
    assert res["targeted_fraction"] < 1.0
    # Causal net profit should exceed blanket profit
    assert res["net_profit"] >= res["blanket_profit"]


def test_roi_curve_coordinates():
    uplifts = np.array([0.04, 0.02, 0.005, -0.01] * 50)
    curve = compute_roi_curve(uplifts, unit_cost=0.10, conversion_value=25.0, n_points=10)

    assert len(curve["fractions"]) == 10
    assert len(curve["cost"]) == 10
    assert len(curve["profit"]) == 10
    # Cost should monotonically increase with fraction
    assert curve["cost"][-1] > curve["cost"][0]


def test_classify_customer_quadrants():
    base_p = np.array([0.08, 0.08, 0.01, 0.02])
    uplifts = np.array([0.02, 0.001, 0.001, -0.01])
    quads = classify_customer_quadrants(base_p, uplifts, prob_threshold=0.05, uplift_threshold=0.01)

    assert quads[0] == "Persuadables (Target Priority)"
    assert quads[1] == "Sure Things (Organic Buyers)"
    assert quads[2] == "Lost Causes (Unreachable)"
    assert quads[3] == "Sleeping Dogs (Do Not Disturb)"
