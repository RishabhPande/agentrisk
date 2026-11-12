"""Calibration contracts for synthetic experiments, not fraud-performance claims."""
import numpy as np
import pandas as pd
import pytest

import run as benchmark


class FixedModel:
    def __init__(self, scores, expected):
        self.scores = np.array(scores)
        self.expected = expected

    def predict_proba(self, frame):
        pd.testing.assert_frame_equal(frame, self.expected)
        return np.column_stack([1 - self.scores, self.scores])


def test_calibration_uses_requested_features_in_order_without_mutation():
    cal = pd.DataFrame({'unused': [9] * 5, 'b': [1, 2, 3, 4, 5],
                        'a': [5, 4, 3, 2, 1]}, index=[8, 3, 7, 1, 4])
    before = cal.copy(deep=True)
    model = FixedModel([0, 0.25, 0.5, 0.75, 1], cal[['a', 'b']].copy())
    hard, step = benchmark.threshold_for(model, ['a', 'b'], cal, rate=0.1)
    assert hard == pytest.approx(0.9)
    assert step == pytest.approx(0.5)
    assert isinstance(hard, float) and isinstance(step, float)
    pd.testing.assert_frame_equal(cal, before)


def test_default_calibration_quantiles_match_documented_operating_point():
    cal = pd.DataFrame({'x': np.arange(101)})
    model = FixedModel(np.linspace(0, 1, 101), cal.copy())
    hard, step = benchmark.threshold_for(model, ['x'], cal)
    assert hard == pytest.approx(0.98)
    assert step == pytest.approx(0.9)


def test_tied_calibration_scores_do_not_promise_an_exact_decline_rate():
    cal = pd.DataFrame({'x': [1, 2, 3, 4]})
    model = FixedModel([0.5] * 4, cal.copy())
    hard, step = benchmark.threshold_for(model, ['x'], cal)
    assert (hard, step) == (0.5, 0.5)
    assert benchmark.decide_ml(np.array([0.499, 0.5, 0.501]), hard).tolist() == [
        'approve', 'decline', 'decline']


@pytest.mark.parametrize('threshold', [0.0, 0.5, 1.0])
def test_ml_threshold_is_inclusive_and_does_not_mutate_scores(threshold):
    scores = np.array([np.nextafter(threshold, -np.inf), threshold,
                       np.nextafter(threshold, np.inf)])
    original = scores.copy()
    assert benchmark.decide_ml(scores, threshold).tolist() == ['approve', 'decline', 'decline']
    np.testing.assert_array_equal(scores, original)
