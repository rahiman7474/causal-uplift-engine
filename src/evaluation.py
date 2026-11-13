"""
Evaluation metrics for causal uplift models: Qini curves, AUUC, and decile performance.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def compute_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_preds: np.ndarray,
    n_bins: int = 100
) -> Dict[str, Any]:
    """
    Computes Qini curve coordinates and Qini coefficient.

    Formula:
        Qini(k) = Y_{1, k} - (Y_{0, k} * N_{1, k} / N_{0, k})
        where Y_{1, k} and Y_{0, k} are cumulative positive outcomes in treatment and control
        among top k proportion of individuals ranked by predicted uplift.
    """
    df = pd.DataFrame({
        "y": y_true,
        "t": treatment,
        "pred": uplift_preds
    }).sort_values(by="pred", ascending=False).reset_index(drop=True)

    n_total = len(df)
    total_t = np.sum(df["t"] == 1)
    total_c = np.sum(df["t"] == 0)

    # Fractions from 0% to 100%
    fractions = np.linspace(0.01, 1.0, n_bins)
    qini_vals = []
    random_qini_vals = []
    cum_uplift_vals = []

    # Overall cumulative treated/control outcomes
    total_treated_y = np.sum(df.loc[df["t"] == 1, "y"])
    total_control_y = np.sum(df.loc[df["t"] == 0, "y"])
    ratio_tc = total_t / max(total_c, 1)

    for frac in fractions:
        k = int(frac * n_total)
        sub = df.iloc[:k]

        n_t_k = np.sum(sub["t"] == 1)
        n_c_k = np.sum(sub["t"] == 0)

        y_t_k = np.sum(sub.loc[sub["t"] == 1, "y"])
        y_c_k = np.sum(sub.loc[sub["t"] == 0, "y"])

        if n_c_k > 0:
            qini = y_t_k - (y_c_k * (n_t_k / n_c_k))
        else:
            qini = 0.0
        qini_vals.append(float(qini))

        # Expected random line
        expected_random = frac * (total_treated_y - total_control_y * ratio_tc)
        random_qini_vals.append(float(expected_random))

        # Empirical Cumulative Uplift rate
        rate_t = (y_t_k / max(n_t_k, 1))
        rate_c = (y_c_k / max(n_c_k, 1))
        cum_uplift = (rate_t - rate_c) * k
        cum_uplift_vals.append(float(cum_uplift))

    # Area Under Curve (AUUC) & Qini Area
    qini_arr = np.array(qini_vals)
    random_arr = np.array(random_qini_vals)
    qini_score = float(np.trapz(qini_arr - random_arr, fractions))
    normalized_qini = float(qini_score / max(abs(np.trapz(qini_arr, fractions)), 1e-6))

    return {
        "fractions": [round(float(f), 4) for f in fractions],
        "qini_values": [round(float(q), 2) for q in qini_vals],
        "random_values": [round(float(r), 2) for r in random_qini_vals],
        "cumulative_uplift": [round(float(u), 2) for u in cum_uplift_vals],
        "qini_score": round(qini_score, 4),
        "normalized_qini": round(normalized_qini, 4),
    }


def compute_decile_table(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_preds: np.ndarray
) -> pd.DataFrame:
    """
    Computes 10 deciles of targeted customers ranked by predicted CATE.
    Validates monotonicity of uplift across deciles.
    """
    df = pd.DataFrame({
        "y": y_true,
        "t": treatment,
        "pred": uplift_preds
    }).sort_values(by="pred", ascending=False).reset_index(drop=True)

    df["decile"] = pd.qcut(df["pred"].rank(method="first", ascending=False), q=10, labels=range(1, 11))

    decile_records = []
    for d, group in df.groupby("decile"):
        n = len(group)
        t_sub = group[group["t"] == 1]
        c_sub = group[group["t"] == 0]

        rate_t = float(t_sub["y"].mean()) if len(t_sub) > 0 else 0.0
        rate_c = float(c_sub["y"].mean()) if len(c_sub) > 0 else 0.0
        uplift = rate_t - rate_c

        decile_records.append({
            "decile": int(d),
            "size": n,
            "mean_pred_uplift": round(float(group["pred"].mean()), 5),
            "treatment_response_rate": round(rate_t, 5),
            "control_response_rate": round(rate_c, 5),
            "empirical_uplift": round(uplift, 5),
            "incremental_conversions": round(uplift * len(t_sub), 1)
        })

    return pd.DataFrame(decile_records)
