"""Keep misspelled experiments from silently generating mislabeled features."""
import copy
import numpy as np
import pandas as pd
import pytest

import sim


@pytest.mark.parametrize('counts', [{'AVV': 2}, {'H': 2, 'AVV': 2}, {'H': 2, 'AVV': 0}])
def test_unknown_classes_rejected_even_after_a_valid_class(counts):
    with pytest.raises(ValueError, match='Unknown session class.*AVV'):
        sim.generate(counts)


@pytest.mark.parametrize('value', [-1, 1.5, '2', True, np.bool_(True), None])
def test_invalid_counts_rejected(value):
    with pytest.raises(ValueError, match='nonnegative integer'):
        sim.generate({'H': value})


@pytest.mark.parametrize('counts', [{}, {'H': 0, 'AU': 0}])
def test_empty_pack_has_clear_error(counts):
    with pytest.raises(ValueError, match='positive count'):
        sim.generate(counts)


@pytest.mark.parametrize('counts', [None, [], [('H', 2)]])
def test_class_counts_must_be_a_mapping(counts):
    with pytest.raises(ValueError, match='must map'):
        sim.generate(counts)


def test_numpy_integer_counts_and_zero_classes_are_supported():
    df = sim.generate({'H': np.int64(3), 'AU': 0}, seed=1)
    assert len(df) == 3
    assert set(df.cls) == {'H'}
    assert (df.y == 0).all()


@pytest.mark.parametrize('key', list(sim.PARAMS))
@pytest.mark.parametrize('value', [-0.1, 1.1, float('nan'), float('inf'), '0.5', True, None])
def test_probability_overrides_are_validated(key, value):
    with pytest.raises(ValueError, match='finite probability'):
        sim.generate({'H': 2}, params={key: value})


def test_unknown_parameter_is_not_silently_ignored():
    with pytest.raises(ValueError, match='Unknown simulation parameter.*p_cloud_agnet'):
        sim.generate({'H': 2}, params={'p_cloud_agnet': 0.1})


@pytest.mark.parametrize('params', [[], False, [('humanize', 0.1)]])
def test_parameters_must_be_a_mapping(params):
    with pytest.raises(ValueError, match='params must map'):
        sim.generate({'H': 2}, params=params)


@pytest.mark.parametrize('probability', [0, 1, np.float64(0.5)])
def test_probability_boundaries_work_without_mutating_inputs(probability):
    counts = {'H': 3, 'AU': 3}
    params = {'humanize': probability, 'p_cloud_agent': probability}
    before = copy.deepcopy((counts, params, sim.PARAMS))
    a = sim.generate(counts, seed=7, params=params)
    b = sim.generate(counts, seed=7, params=params)
    pd.testing.assert_frame_equal(a, b)
    assert (counts, params, sim.PARAMS) == before
