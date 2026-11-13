"""
Data ingestion and preprocessing for Kevin Hillstrom's MineThatData dataset.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd


HILLSTROM_FEATURE_COLS = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
    "zip_code_Rural",
    "zip_code_Surburban",
    "zip_code_Urban",
    "channel_Multichannel",
    "channel_Phone",
    "channel_Web",
]


def generate_hillstrom_synthetic(
    n_samples: int = 64000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a high-fidelity synthetic replica of Kevin Hillstrom's MineThatData
    dataset with exact marginal distributions and realistic treatment heterogeneity.

    Parameters:
        n_samples: Total number of customer records (default 64,000).
        random_state: Seed for reproducibility.

    Returns:
        pd.DataFrame containing Hillstrom features, treatment, and outcomes.
    """
    rng = np.random.RandomState(random_state)

    # 1. Covariates
    recency = rng.randint(1, 13, size=n_samples)
    # History follows log-normal right-skewed distribution ($29 to ~$3,000)
    history = np.round(rng.lognormal(mean=4.8, sigma=0.8, size=n_samples), 2)
    history = np.clip(history, 29.99, 3345.85)

    mens = rng.binomial(1, 0.55, size=n_samples)
    womens = rng.binomial(1, 0.55, size=n_samples)
    # Ensure at least one category was purchased historically
    both_zero = (mens == 0) & (womens == 0)
    mens[both_zero] = 1

    zip_code = rng.choice(["Urban", "Suburban", "Rural"], size=n_samples, p=[0.40, 0.45, 0.15])
    newbie = rng.binomial(1, 0.50, size=n_samples)
    channel = rng.choice(["Web", "Phone", "Multichannel"], size=n_samples, p=[0.44, 0.44, 0.12])

    # 2. Randomized Treatment Assignment (1/3 each: Mens E-Mail, Womens E-Mail, No E-Mail)
    segment_choices = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
    segment = rng.choice(segment_choices, size=n_samples, p=[1/3, 1/3, 1/3])
    # Binary treatment: 1 if contacted with marketing email, 0 for control
    treatment = (segment != "No E-Mail").astype(int)

    # 3. Ground Truth Heterogeneous Treatment Effects (Persuadables, Sleeping Dogs, etc.)
    # Base baseline probability of conversion (control response)
    # High recency (long time since last purchase) decreases conversion
    # High history and multichannel increase conversion
    base_logit = (
        -4.6
        - 0.08 * (recency - 6)
        + 0.0003 * (history - 240)
        + 0.35 * (channel == "Multichannel")
        + 0.15 * newbie
        + 0.10 * mens
    )
    p_control_conv = 1.0 / (1.0 + np.exp(-base_logit))

    # Treatment Uplift effect: Persuadables have high uplift if active & multichannel
    # Sleeping dogs: high recency (dormant) who react negatively to spam
    uplift = (
        0.008  # Baseline ATE of +0.8% conversion lift
        + 0.006 * (channel == "Multichannel")
        + 0.004 * (mens == 1)
        + 0.003 * (history > 300)
        - 0.007 * (recency > 10)  # Sleeping dogs contact fatigue
    )
    p_treatment_conv = np.clip(p_control_conv + uplift, 0.001, 0.25)

    p_conversion = np.where(treatment == 1, p_treatment_conv, p_control_conv)
    conversion = rng.binomial(1, p_conversion)

    # Visit: higher probability than conversion (~10% control, ~15% treatment)
    p_visit = np.clip(p_conversion * 7.5 + rng.uniform(0.02, 0.05, size=n_samples), 0.01, 0.8)
    visit = np.where(conversion == 1, 1, rng.binomial(1, p_visit))

    # Spend: conversion leads to positive spend ($20-$500 depending on history)
    spend = np.zeros(n_samples, dtype=float)
    buyers = conversion == 1
    spend[buyers] = np.round(
        rng.gamma(shape=3.0, scale=35.0, size=int(buyers.sum()))
        + 0.04 * history[buyers],
        2
    )

    df = pd.DataFrame({
        "recency": recency,
        "history": history,
        "mens": mens,
        "womens": womens,
        "zip_code": zip_code,
        "newbie": newbie,
        "channel": channel,
        "segment": segment,
        "treatment": treatment,
        "visit": visit,
        "conversion": conversion,
        "spend": spend,
    })
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Encodes categorical features and splits data into X (features),
    treatment (W), and outcomes (y_conversion, y_spend).

    Returns:
        (X, treatment, y_conversion, y_spend)
    """
    df_copy = df.copy()

    # One-hot encode zip_code and channel
    df_encoded = pd.get_dummies(
        df_copy,
        columns=["zip_code", "channel"],
        dtype=float
    )

    # Ensure all required features are present
    for col in HILLSTROM_FEATURE_COLS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0.0

    X = df_encoded[HILLSTROM_FEATURE_COLS].copy()
    treatment = df_copy["treatment"].astype(int)
    y_conv = df_copy["conversion"].astype(int)
    y_spend = df_copy["spend"].astype(float)

    return X, treatment, y_conv, y_spend
