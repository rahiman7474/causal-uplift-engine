"""
Marginal ROI Optimizer and budget-constrained uplift targeting engine.
Transforms predicted CATE into optimal business targeting decisions.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


def optimize_targeting_policy(
    uplift_preds: np.ndarray,
    unit_cost: float = 0.15,
    conversion_value: float = 35.0,
    budget: Optional[float] = None
) -> Dict[str, Any]:
    """
    Finds the optimal customer targeting cut-off to maximize expected net profit.

    Net Profit for customer i:
        Profit_i = tau_i * conversion_value - unit_cost

    Optimal unconstrained threshold:
        tau* = unit_cost / conversion_value

    Parameters:
        uplift_preds: Array of predicted CATE conversion uplifts (tau_i).
        unit_cost: Cost to contact/treat a single customer (in dollars).
        conversion_value: Gross dollar margin generated per incremental conversion.
        budget: Maximum spend allowed (optional).

    Returns:
        Dictionary with optimal threshold, targeted count, expected incremental conversions,
        spend, gross revenue, net profit, and ROI.
    """
    n_customers = len(uplift_preds)
    threshold = unit_cost / max(conversion_value, 1e-4)

    # Sort descending by predicted uplift
    sorted_indices = np.argsort(uplift_preds)[::-1]
    sorted_uplifts = uplift_preds[sorted_indices]

    # Calculate cumulative metrics across all potential cutoffs
    cum_customers = np.arange(1, n_customers + 1)
    cum_uplift = np.cumsum(sorted_uplifts)
    cum_cost = cum_customers * unit_cost
    cum_revenue = cum_uplift * conversion_value
    cum_profit = cum_revenue - cum_cost

    # If budget is constrained, find max index allowed by budget
    if budget is not None and budget > 0:
        max_allowed_customers = int(budget / unit_cost)
        valid_range = slice(0, min(max_allowed_customers, n_customers))
    else:
        valid_range = slice(0, n_customers)

    # Optimal cutoff is argmax of cumulative profit within budget
    profits_in_range = cum_profit[valid_range]
    if len(profits_in_range) > 0:
        best_k = int(np.argmax(profits_in_range)) + 1
    else:
        best_k = 0

    optimal_tau = float(sorted_uplifts[best_k - 1]) if best_k > 0 else float(threshold)
    targeted_mask = (uplift_preds >= optimal_tau)
    targeted_count = int(targeted_mask.sum())

    total_incremental_conv = float(uplift_preds[targeted_mask].sum()) if targeted_count > 0 else 0.0
    total_cost = targeted_count * unit_cost
    gross_revenue = total_incremental_conv * conversion_value
    net_profit = gross_revenue - total_cost
    roi_pct = (net_profit / max(total_cost, 1e-4)) * 100.0

    # Blanket targeting baseline (targeting 100% of customers)
    blanket_uplift = float(uplift_preds.sum())
    blanket_cost = n_customers * unit_cost
    blanket_revenue = blanket_uplift * conversion_value
    blanket_profit = blanket_revenue - blanket_cost
    blanket_roi = (blanket_profit / max(blanket_cost, 1e-4)) * 100.0

    profit_lift_vs_blanket = net_profit - blanket_profit
    budget_saved = blanket_cost - total_cost

    return {
        "unit_cost": unit_cost,
        "conversion_value": conversion_value,
        "optimal_threshold_tau": round(optimal_tau, 6),
        "total_population": n_customers,
        "targeted_customers": targeted_count,
        "targeted_fraction": round(targeted_count / n_customers, 4),
        "expected_incremental_conversions": round(total_incremental_conv, 1),
        "campaign_cost": round(total_cost, 2),
        "gross_revenue": round(gross_revenue, 2),
        "net_profit": round(net_profit, 2),
        "roi_pct": round(roi_pct, 2),
        "blanket_profit": round(blanket_profit, 2),
        "blanket_roi": round(blanket_roi, 2),
        "profit_lift_vs_blanket": round(profit_lift_vs_blanket, 2),
        "budget_saved": round(budget_saved, 2),
    }


def compute_roi_curve(
    uplift_preds: np.ndarray,
    unit_cost: float = 0.15,
    conversion_value: float = 35.0,
    n_points: int = 100
) -> Dict[str, List[float]]:
    """
    Computes cumulative profit curve coordinates for interactive plotting.
    """
    n = len(uplift_preds)
    sorted_uplifts = np.sort(uplift_preds)[::-1]

    fractions = np.linspace(0.01, 1.0, n_points)
    k_vals = (fractions * n).astype(int)

    cum_uplifts = np.cumsum(sorted_uplifts)
    curve_fractions = []
    curve_cost = []
    curve_revenue = []
    curve_profit = []

    for f, k in zip(fractions, k_vals):
        k = max(1, min(k, n))
        inc_conv = float(cum_uplifts[k - 1])
        c = float(k * unit_cost)
        rev = float(inc_conv * conversion_value)
        prof = rev - c

        curve_fractions.append(round(float(f), 4))
        curve_cost.append(round(c, 2))
        curve_revenue.append(round(rev, 2))
        curve_profit.append(round(prof, 2))

    return {
        "fractions": curve_fractions,
        "cost": curve_cost,
        "revenue": curve_revenue,
        "profit": curve_profit
    }


def classify_customer_quadrants(
    base_probabilities: np.ndarray,
    uplift_preds: np.ndarray,
    prob_threshold: float = 0.05,
    uplift_threshold: float = 0.005
) -> np.ndarray:
    """
    Classifies customers into standard 4 marketing quadrants:
    1. Persuadables: High uplift (uplift > uplift_threshold)
    2. Sure Things: High base conversion, low uplift (base > prob_threshold, uplift <= uplift_threshold and >= 0)
    3. Lost Causes: Low base, low uplift (base <= prob_threshold, uplift >= 0)
    4. Sleeping Dogs: Negative uplift (uplift < 0)
    """
    quadrants = np.empty(len(uplift_preds), dtype=object)

    for i in range(len(uplift_preds)):
        u = uplift_preds[i]
        b = base_probabilities[i]

        if u < 0:
            quadrants[i] = "Sleeping Dogs (Do Not Disturb)"
        elif u >= uplift_threshold:
            quadrants[i] = "Persuadables (Target Priority)"
        elif b >= prob_threshold:
            quadrants[i] = "Sure Things (Organic Buyers)"
        else:
            quadrants[i] = "Lost Causes (Unreachable)"

    return quadrants
