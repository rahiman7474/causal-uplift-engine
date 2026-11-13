"""
Causal Refutation and Sensitivity Tests.
Validates model robustness via Placebo Treatment and Random Common Cause refutations.
"""

from typing import Dict, Any, Type
import numpy as np
import pandas as pd


def placebo_treatment_refutation(
    model_cls: Type,
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.Series,
    n_simulations: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Placebo Treatment Refutation:
    Replaces the true treatment assignment vector with a randomly permuted treatment vector.
    Under the null hypothesis of no confounding in placebo, the estimated average CATE
    must collapse to zero (tau_placebo ~ 0).

    Returns:
        Dict with original effect, mean placebo effect, p-value, and pass status.
    """
    rng = np.random.RandomState(random_state)

    # 1. Fit baseline model to get original estimated ATE
    base_model = model_cls(n_estimators=40, random_state=random_state)
    base_model.fit(X, treatment, y)
    original_uplift = base_model.predict_uplift(X)
    original_ate = float(np.mean(original_uplift))

    # 2. Run Placebo refutations
    placebo_ates = []
    t_values = np.asarray(treatment)

    for i in range(n_simulations):
        permuted_t = pd.Series(rng.permutation(t_values), index=treatment.index)
        placebo_model = model_cls(n_estimators=40, random_state=random_state + i + 1)
        placebo_model.fit(X, permuted_t, y)
        pred_placebo = placebo_model.predict_uplift(X)
        placebo_ates.append(float(np.mean(pred_placebo)))

    mean_placebo_ate = float(np.mean(placebo_ates))
    std_placebo_ate = float(np.std(placebo_ates)) if len(placebo_ates) > 1 else 1e-4

    # Two-tailed p-value: probability of observing effect as extreme as original under placebo distribution
    z_score = abs(original_ate - mean_placebo_ate) / max(std_placebo_ate, 1e-6)
    # Refutation passes if placebo effect is near 0 (< 20% of original effect)
    refutation_passed = bool(abs(mean_placebo_ate) < 0.20 * abs(original_ate))

    return {
        "test_name": "Placebo Treatment Refutation",
        "original_ate": round(original_ate, 6),
        "mean_placebo_ate": round(mean_placebo_ate, 6),
        "placebo_simulations": [round(p, 6) for p in placebo_ates],
        "refutation_passed": refutation_passed,
        "z_score": round(float(z_score), 2),
        "conclusion": (
            "PASSED: Estimated effect collapses towards zero when treatment is randomized."
            if refutation_passed
            else "WARNING: Model captured non-zero effect on placebo treatment."
        )
    }


def random_common_cause_refutation(
    model_cls: Type,
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.Series,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Random Common Cause Refutation:
    Adds an independent random normal noise variable to X.
    The estimated causal effect should remain statistically invariant (drift < 10%).
    """
    rng = np.random.RandomState(random_state)

    # 1. Original model
    base_model = model_cls(n_estimators=40, random_state=random_state)
    base_model.fit(X, treatment, y)
    orig_ate = float(np.mean(base_model.predict_uplift(X)))

    # 2. Model with pure noise covariate
    X_noise = X.copy()
    X_noise["random_noise_covariate"] = rng.normal(0, 1, size=len(X))

    noisy_model = model_cls(n_estimators=40, random_state=random_state)
    noisy_model.fit(X_noise, treatment, y)
    noisy_ate = float(np.mean(noisy_model.predict_uplift(X_noise)))

    drift_pct = abs(noisy_ate - orig_ate) / max(abs(orig_ate), 1e-6) * 100.0
    passed = bool(drift_pct < 10.0)

    return {
        "test_name": "Random Common Cause Refutation",
        "original_ate": round(orig_ate, 6),
        "noisy_ate": round(noisy_ate, 6),
        "drift_pct": round(drift_pct, 2),
        "refutation_passed": passed,
        "conclusion": (
            f"PASSED: Effect drift with noise covariate is {round(drift_pct, 2)}% (< 10%)."
            if passed
            else "WARNING: Causal estimate showed sensitivity to noise variable."
        )
    }
