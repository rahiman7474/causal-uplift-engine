"""
Covariate balance diagnostics and Standardized Mean Difference (SMD) validation.
Checks unconfoundedness and common support (positivity) assumptions.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def calculate_smd(
    X: pd.DataFrame,
    treatment: pd.Series
) -> pd.DataFrame:
    """
    Computes Standardized Mean Difference (SMD) for all covariates between
    Treatment (W=1) and Control (W=0) cohorts.

    Formula:
        SMD = |mean(X_1) - mean(X_0)| / sqrt((var(X_1) + var(X_0)) / 2)

    Standard Benchmark:
        - SMD < 0.10: Excellent balance (no confounding).
        - 0.10 <= SMD <= 0.20: Marginal balance.
        - SMD > 0.20: Imbalance requiring propensity adjustment / IPW.
    """
    t_mask = (treatment == 1)
    c_mask = (treatment == 0)

    records = []
    for col in X.columns:
        x_t = X.loc[t_mask, col].values
        x_c = X.loc[c_mask, col].values

        mean_t = float(np.mean(x_t))
        mean_c = float(np.mean(x_c))

        var_t = float(np.var(x_t, ddof=1)) if len(x_t) > 1 else 0.0
        var_c = float(np.var(x_c, ddof=1)) if len(x_c) > 1 else 0.0

        pooled_std = np.sqrt((var_t + var_c) / 2.0)
        if pooled_std < 1e-12:
            smd = 0.0
        else:
            smd = abs(mean_t - mean_c) / pooled_std

        status = "Balanced" if smd < 0.10 else ("Marginal" if smd <= 0.20 else "Imbalanced")

        records.append({
            "feature": col,
            "mean_treatment": round(mean_t, 4),
            "mean_control": round(mean_c, 4),
            "smd": round(float(smd), 4),
            "status": status,
        })

    report_df = pd.DataFrame(records).sort_values(by="smd", ascending=False)
    return report_df


def check_overlap_and_positivity(
    X: pd.DataFrame,
    treatment: pd.Series
) -> Dict[str, Any]:
    """
    Evaluates the Positivity (Common Support) assumption by fitting a propensity model
    and examining min/max propensity scores across both treatment groups.

    Positivity requires 0 < P(W=1|X) < 1 across all regions of covariate space X.
    """
    propensity_model = LogisticRegression(max_iter=1000, random_state=42)
    propensity_model.fit(X, treatment)
    propensities = propensity_model.predict_proba(X)[:, 1]

    t_mask = (treatment == 1)
    c_mask = (treatment == 0)

    t_props = propensities[t_mask]
    c_props = propensities[c_mask]

    min_overlap = max(float(np.min(t_props)), float(np.min(c_props)))
    max_overlap = min(float(np.max(t_props)), float(np.max(c_props)))

    # Positivity violation check: trimming thresholds 0.01 and 0.99
    extreme_low = float(np.mean(propensities < 0.02))
    extreme_high = float(np.mean(propensities > 0.98))
    positivity_satisfied = bool((extreme_low < 0.01) and (extreme_high < 0.01))

    return {
        "positivity_satisfied": positivity_satisfied,
        "propensity_mean": round(float(np.mean(propensities)), 4),
        "treatment_prop_range": [round(float(np.min(t_props)), 4), round(float(np.max(t_props)), 4)],
        "control_prop_range": [round(float(np.min(c_props)), 4), round(float(np.max(c_props)), 4)],
        "overlap_window": [round(min_overlap, 4), round(max_overlap, 4)],
        "extreme_propensity_pct": round((extreme_low + extreme_high) * 100, 2),
    }
