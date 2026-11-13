import pytest
import numpy as np
import pandas as pd
from src.data import generate_hillstrom_synthetic, preprocess_data
from src.models import TLearner, SLearner, XLearner, DoubleMachineLearning


@pytest.fixture(scope="module")
def sample_data():
    df = generate_hillstrom_synthetic(n_samples=600, random_state=42)
    return preprocess_data(df)


def test_t_learner(sample_data):
    X, treatment, y_conv, _ = sample_data
    model = TLearner(n_estimators=15, max_depth=2, random_state=42)
    model.fit(X, treatment, y_conv)
    uplift = model.predict_uplift(X)

    assert len(uplift) == len(X)
    assert not np.isnan(uplift).any()


def test_s_learner(sample_data):
    X, treatment, y_conv, _ = sample_data
    model = SLearner(n_estimators=15, max_depth=2, random_state=42)
    model.fit(X, treatment, y_conv)
    uplift = model.predict_uplift(X)

    assert len(uplift) == len(X)
    assert not np.isnan(uplift).any()


def test_x_learner(sample_data):
    X, treatment, y_conv, _ = sample_data
    model = XLearner(n_estimators=15, max_depth=2, random_state=42)
    model.fit(X, treatment, y_conv)
    uplift = model.predict_uplift(X)

    assert len(uplift) == len(X)
    assert not np.isnan(uplift).any()


def test_double_machine_learning(sample_data):
    X, treatment, y_conv, _ = sample_data
    model = DoubleMachineLearning(n_splits=2, n_estimators=15, max_depth=2, random_state=42)
    model.fit(X, treatment, y_conv)
    uplift = model.predict_uplift(X)

    assert len(uplift) == len(X)
    assert not np.isnan(uplift).any()
