"""
Causal Uplift Meta-Learners (T-Learner, S-Learner, X-Learner) and Double Machine Learning (DML).
Estimates Conditional Average Treatment Effects (CATE): tau(x) = E[Y(1) - Y(0) | X = x].
"""

from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
import lightgbm as lgb


class TLearner:
    """
    Two-Model Learner (T-Learner):
    Fits separate estimators for the treatment group and control group.
    tau(X) = mu_1(X) - mu_0(X)
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_treatment: Optional[lgb.LGBMRegressor] = None
        self.model_control: Optional[lgb.LGBMRegressor] = None

    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series):
        X_arr = np.asarray(X)
        t_arr = np.asarray(treatment)
        y_arr = np.asarray(y)

        t_mask = (t_arr == 1)
        c_mask = (t_arr == 0)

        self.model_treatment = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=-1
        )
        self.model_control = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state + 1,
            verbose=-1
        )

        self.model_treatment.fit(X_arr[t_mask], y_arr[t_mask])
        self.model_control.fit(X_arr[c_mask], y_arr[c_mask])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = np.asarray(X)
        mu_1 = self.model_treatment.predict(X_arr)
        mu_0 = self.model_control.predict(X_arr)
        return mu_1 - mu_0


class SLearner:
    """
    Single-Model Learner (S-Learner):
    Fits a single estimator with treatment indicator W as a feature.
    tau(X) = mu(X, 1) - mu(X, 0)
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model: Optional[lgb.LGBMRegressor] = None

    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series):
        X_arr = np.asarray(X)
        t_arr = np.asarray(treatment).reshape(-1, 1)
        X_with_t = np.hstack([X_arr, t_arr])
        y_arr = np.asarray(y)

        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=-1
        )
        self.model.fit(X_with_t, y_arr)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = np.asarray(X)
        n = X_arr.shape[0]
        X_treat = np.hstack([X_arr, np.ones((n, 1))])
        X_ctrl = np.hstack([X_arr, np.zeros((n, 1))])
        return self.model.predict(X_treat) - self.model.predict(X_ctrl)


class XLearner:
    """
    Cross-Learner (X-Learner; Künzel et al. 2019):
    Specifically engineered for unbalanced treatment assignment ratios.
    1. Estimate mu_0, mu_1 using base learners.
    2. Impute counterfactual effects D_1 = Y_1 - mu_0(X_1) and D_0 = mu_1(X_0) - Y_0.
    3. Train tau_1(X) on treated and tau_0(X) on control.
    4. Combine: tau(X) = e(X)*tau_0(X) + (1 - e(X))*tau_1(X).
    """
    def __init__(self, n_estimators: int = 80, max_depth: int = 4, learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.stage1_t: Optional[lgb.LGBMRegressor] = None
        self.stage1_c: Optional[lgb.LGBMRegressor] = None
        self.stage2_t: Optional[lgb.LGBMRegressor] = None
        self.stage2_c: Optional[lgb.LGBMRegressor] = None
        self.propensity_score: float = 0.5

    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series):
        X_arr = np.asarray(X)
        t_arr = np.asarray(treatment)
        y_arr = np.asarray(y)

        t_mask = (t_arr == 1)
        c_mask = (t_arr == 0)
        self.propensity_score = float(np.mean(t_mask))

        # Stage 1: Base response surfaces
        self.stage1_t = lgb.LGBMRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, random_state=self.random_state, verbose=-1
        ).fit(X_arr[t_mask], y_arr[t_mask])

        self.stage1_c = lgb.LGBMRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, random_state=self.random_state + 1, verbose=-1
        ).fit(X_arr[c_mask], y_arr[c_mask])

        # Stage 2: Impute counterfactuals
        # For treated: actual Y_1 minus predicted control counterfactual
        d_1 = y_arr[t_mask] - self.stage1_c.predict(X_arr[t_mask])
        # For control: predicted treatment counterfactual minus actual Y_0
        d_0 = self.stage1_t.predict(X_arr[c_mask]) - y_arr[c_mask]

        self.stage2_t = lgb.LGBMRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, random_state=self.random_state + 2, verbose=-1
        ).fit(X_arr[t_mask], d_1)

        self.stage2_c = lgb.LGBMRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, random_state=self.random_state + 3, verbose=-1
        ).fit(X_arr[c_mask], d_0)

        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = np.asarray(X)
        pred_tau_t = self.stage2_t.predict(X_arr)
        pred_tau_c = self.stage2_c.predict(X_arr)
        e = self.propensity_score
        return e * pred_tau_c + (1.0 - e) * pred_tau_t


class DoubleMachineLearning:
    """
    Double Machine Learning (DML; Chernozhukov et al. 2018):
    Uses Neyman-orthogonal score and Robinson residualization to purge confounding
    and regularization bias via K-fold cross-fitting.
    
    Residualize Y: Y_tilde = Y - E[Y|X]
    Residualize W: W_tilde = W - E[W|X]
    CATE is estimated by regressing Y_tilde on W_tilde interacting with X.
    """
    def __init__(self, n_splits: int = 3, n_estimators: int = 80, max_depth: int = 4, random_state: int = 42):
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.final_cate_model: Optional[lgb.LGBMRegressor] = None

    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series):
        X_arr = np.asarray(X)
        t_arr = np.asarray(treatment).astype(float)
        y_arr = np.asarray(y).astype(float)

        n = len(X_arr)
        y_res = np.zeros(n)
        t_res = np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(X_arr):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
            t_tr, t_val = t_arr[train_idx], t_arr[val_idx]

            # Model outcome nuisance E[Y|X]
            m_y = lgb.LGBMRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, verbose=-1, random_state=self.random_state)
            m_y.fit(X_tr, y_tr)
            y_res[val_idx] = y_val - m_y.predict(X_val)

            # Model propensity nuisance E[W|X]
            m_w = lgb.LGBMRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, verbose=-1, random_state=self.random_state + 1)
            m_w.fit(X_tr, t_tr)
            t_res[val_idx] = t_val - m_w.predict(X_val)

        # Orthogonalized target: Y_res / T_res with sample weights T_res^2
        # Ensures numerical stability by clipping very small denominators
        denom = np.where(np.abs(t_res) < 0.1, np.sign(t_res) * 0.1, t_res)
        pseudo_outcome = y_res / denom
        weights = t_res ** 2

        self.final_cate_model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            random_state=self.random_state + 2,
            verbose=-1
        )
        self.final_cate_model.fit(X_arr, pseudo_outcome, sample_weight=weights)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = np.asarray(X)
        return self.final_cate_model.predict(X_arr)
