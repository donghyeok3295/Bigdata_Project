import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.linear_model import LogisticRegression

from src.models import metrics as m


@pytest.fixture
def scored():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 5000)
    y = (rng.random(5000) < p).astype(int)
    return y, p


def test_match_sklearn(scored):
    y, p = scored
    assert m.brier(y, p) == pytest.approx(skm.brier_score_loss(y, p))
    assert m.log_loss(y, p) == pytest.approx(skm.log_loss(y, p))
    assert m.auc(y, p) == pytest.approx(skm.roc_auc_score(y, p))
    assert m.average_precision(y, p) == pytest.approx(skm.average_precision_score(y, p))


def test_auc_ap_with_ties():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.2, 0.2, 0.5, 0.5, 0.9, 0.1])
    assert m.auc(y, p) == pytest.approx(skm.roc_auc_score(y, p))
    assert m.average_precision(y, p) == pytest.approx(skm.average_precision_score(y, p))


def test_slope_matches_logistic_fit_and_near_one_when_calibrated(scored):
    y, p = scored
    slope, intercept = m.calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.1)
    assert intercept == pytest.approx(0.0, abs=0.1)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    ref = LogisticRegression(C=1e10).fit(x, y)
    assert slope == pytest.approx(ref.coef_[0, 0], abs=1e-3)
    assert intercept == pytest.approx(ref.intercept_[0], abs=1e-3)


def test_slope_below_one_when_overconfident(scored):
    y, p = scored
    logit = np.log(p / (1 - p)) * 2
    over = 1 / (1 + np.exp(-logit))
    assert m.calibration_slope_intercept(y, over)[0] == pytest.approx(0.5, abs=0.1)


def test_log_loss_clips_extremes():
    assert np.isfinite(m.log_loss(np.array([1, 0]), np.array([0.0, 1.0])))


def test_block_bootstrap_reproducible_and_block_based(scored):
    y, p = scored
    dates = np.repeat(np.arange(50), 100)
    a = m.block_bootstrap(y, p, dates, m.brier, 20, seed=42)
    b = m.block_bootstrap(y, p, dates, m.brier, 20, seed=42)
    assert np.array_equal(a, b)
    assert a.std() > 0


def test_citl_zero_when_calibrated_and_sign_follows_bias(scored):
    y, p = scored
    assert m.calibration_in_the_large(y, p) == pytest.approx(0.0, abs=0.1)
    logit = np.log(p / (1 - p))
    biased_up = 1 / (1 + np.exp(-(logit + 1.0)))
    biased_down = 1 / (1 + np.exp(-(logit - 1.0)))
    assert m.calibration_in_the_large(y, biased_up) == pytest.approx(-1.0, abs=0.15)
    assert m.calibration_in_the_large(y, biased_down) == pytest.approx(1.0, abs=0.15)
    assert "citl" in m.all_metrics(y, p)
