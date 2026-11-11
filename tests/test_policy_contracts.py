"""Exact policy regression contracts, not estimates of production fraud performance."""
import numpy as np
import pandas as pd
import pytest

import run as benchmark
import sim


def sessions(n=1):
    """Verified, bound, in-scope synthetic principals with no automation flag."""
    return pd.DataFrame({
        'webdriver': np.zeros(n), 'untrusted_ratio': np.zeros(n),
        'headless_ua': np.zeros(n), 'keydown_per_input': np.ones(n),
        'mousemove': np.full(n, 100), 'is_mobile': np.zeros(n),
        'sig_present': np.ones(n), 'sig_valid': np.ones(n),
        'mandate_present': np.ones(n), 'mandate_valid': np.ones(n),
        'cred_bound_to_principal': np.ones(n), 'merchant_in_scope': np.ones(n),
        'category_in_scope': np.ones(n), 'amount_over_cap': np.zeros(n),
        'ship_to_principal': np.ones(n), 'y': np.zeros(n), 'cls': ['AV'] * n,
    })


@pytest.mark.parametrize('field', ['sig_valid', 'mandate_valid'])
@pytest.mark.parametrize('score', [0.1, 0.6, 0.9])
def test_present_invalid_credential_always_hard_declines(field, score):
    df = sessions()
    df[field] = 0
    assert benchmark.apply_gate(df, np.array([score]), 0.8, 0.4).tolist() == ['decline']


@pytest.mark.parametrize('field,value', [
    ('cred_bound_to_principal', 0), ('merchant_in_scope', 0),
    ('category_in_scope', 0), ('ship_to_principal', 0),
    ('amount_over_cap', benchmark.SCOPE_TOL + 0.001),
])
def test_verified_binding_or_scope_violation_steps_up_without_overriding_decline(field, value):
    df = sessions(2)
    df[field] = value
    assert benchmark.apply_gate(df, np.array([0.1, 0.8]), 0.8, 0.4).tolist() == ['stepup', 'decline']


def test_amount_tolerance_boundary_is_inclusive_and_inputs_unchanged():
    df = sessions(2)
    df['amount_over_cap'] = [benchmark.SCOPE_TOL, benchmark.SCOPE_TOL + 0.001]
    before = df.copy(deep=True)
    score = np.array([0.1, 0.1])
    assert benchmark.apply_gate(df, score, 0.8, 0.4).tolist() == ['approve', 'stepup']
    pd.testing.assert_frame_equal(df, before)
    np.testing.assert_array_equal(score, [0.1, 0.1])


def test_unattested_automation_grey_band_and_decline_boundaries():
    df = sessions(4)
    df['sig_present'] = 0
    df['sig_valid'] = 0
    df['webdriver'] = 1
    assert benchmark.apply_gate(df, np.array([0.399, 0.4, 0.799, 0.8]), 0.8, 0.4).tolist() == [
        'approve', 'stepup', 'stepup', 'decline']


def test_human_or_verified_in_scope_grey_band_is_not_stepped_up():
    df = sessions(2)
    df.loc[0, ['sig_present', 'sig_valid', 'mandate_present', 'mandate_valid']] = 0
    assert benchmark.apply_gate(df, np.array([0.6, 0.6]), 0.8, 0.4).tolist() == ['approve', 'approve']


@pytest.mark.parametrize('field,value', [
    ('webdriver', 1), ('untrusted_ratio', 0.1), ('headless_ua', 1),
])
def test_explicit_automation_signals_are_flagged(field, value):
    df = sessions()
    df[field] = value
    assert benchmark.automation_flag(df).tolist() == [1]


def test_low_mouse_keyboard_flag_exempts_mobile_and_respects_strict_boundaries():
    df = sessions(4)
    df['keydown_per_input'] = [0.049, 0.05, 0.049, 0.049]
    df['mousemove'] = [19, 19, 20, 19]
    df['is_mobile'] = [0, 0, 0, 1]
    assert benchmark.automation_flag(df).tolist() == [1, 0, 0, 0]


def test_stepup_resolves_only_challenges_with_exact_class_probabilities():
    class FixedDraws:
        def random(self, n):
            assert n == 8
            return np.array([0.919, 0.92, 0.099, 0.10, 0.049, 0.05, 0, 0.99])
    df = sessions(8)
    df['y'] = [0, 0, 1, 1, 1, 1, 0, 1]
    df['cls'] = ['AV', 'H', 'FHJ', 'FHJ', 'FB', 'FAV', 'AU', 'FSP']
    decisions = np.array(['stepup'] * 6 + ['decline', 'approve'])
    original = decisions.copy()
    assert benchmark.resolve_stepup(df, decisions, FixedDraws()).tolist() == [
        'approve', 'decline', 'approve', 'decline', 'approve', 'decline', 'decline', 'approve']
    np.testing.assert_array_equal(decisions, original)


def test_per_class_separates_prechallenge_and_final_outcomes():
    df = pd.DataFrame({'cls': ['H', 'H', 'AV', 'AV']})
    pre = np.array(['approve', 'stepup', 'decline', 'stepup'])
    final = np.array(['approve', 'approve', 'decline', 'decline'])
    assert benchmark.per_class(df, pre, final) == {
        'H': {'decline': 0.0, 'stepup': 0.5, 'hard_decline': 0.0},
        'AV': {'decline': 1.0, 'stepup': 0.5, 'hard_decline': 0.5},
    }


def test_generated_all_class_gate_decisions_preserve_hard_rejections():
    df = sim.generate({c: 50 for c in sim.ALL}, seed=13)
    scores = np.random.default_rng(3).random(len(df))
    pre = benchmark.apply_gate(df, scores, 0.8, 0.4)
    final = benchmark.resolve_stepup(df, pre, np.random.default_rng(7))
    assert set(pre) <= {'approve', 'stepup', 'decline'}
    assert set(final) <= {'approve', 'decline'}
    assert (final[pre == 'decline'] == 'decline').all()
    assert (pre[(df.sig_present == 1) & (df.sig_valid == 0)] == 'decline').all()
