from __future__ import annotations
from unittest import mock
import numpy as np
import pytest
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study._study_direction import StudyDirection
from optuna.terminator.improvement import _preprocessing
from optuna.trial import create_trial
from optuna.trial import FrozenTrial

def _get_trial_values(trials: list[FrozenTrial]) -> list[float]:
    if False:
        for i in range(10):
            print('nop')
    values = []
    for t in trials:
        assert t.value is not None
        values.append(t.value)
    return values

@pytest.mark.parametrize('direction', (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_preprocessing_pipeline(direction: StudyDirection) -> None:
    if False:
        print('Hello World!')
    p1 = _preprocessing.NullPreprocessing()
    p2 = _preprocessing.NullPreprocessing()
    pipeline = _preprocessing.PreprocessingPipeline([p1, p2])
    with mock.patch.object(p1, 'apply') as a1:
        with mock.patch.object(p2, 'apply') as a2:
            pipeline.apply([], direction)
            a1.assert_called_once()
            a2.assert_called_once()

@pytest.mark.parametrize('direction', (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_null_preprocessing(direction: StudyDirection) -> None:
    if False:
        i = 10
        return i + 15
    trials_before = [create_trial(params={'x': 0.0}, distributions={'x': FloatDistribution(-1, 1)}, value=1.0)]
    p = _preprocessing.NullPreprocessing()
    trials_after = p.apply(trials_before, direction)
    assert trials_before == trials_after

@pytest.mark.parametrize('direction', (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_unscale_log(direction: StudyDirection) -> None:
    if False:
        return 10
    trials_before = [create_trial(params={'categorical': 0, 'float': 1, 'float_log': 2, 'int': 3, 'int_log': 4}, distributions={'categorical': CategoricalDistribution((0, 1, 2)), 'float': FloatDistribution(1, 10), 'float_log': FloatDistribution(1, 10, log=True), 'int': IntDistribution(1, 10), 'int_log': IntDistribution(1, 10, log=True)}, value=1.0)]
    p = _preprocessing.UnscaleLog()
    trials_after = p.apply(trials_before, direction)
    assert len(trials_before) == len(trials_after) == 1
    assert trials_before[0].value == trials_after[0].value
    for d in trials_after[0].distributions:
        if isinstance(d, (IntDistribution, FloatDistribution)):
            assert not d.log
    for name in ['float', 'int', 'categorical']:
        assert trials_before[0].params[name] == trials_after[0].params[name]
        assert trials_before[0].distributions[name] == trials_after[0].distributions[name]
    for name in ['float_log', 'int_log']:
        assert np.log(trials_before[0].params[name]) == trials_after[0].params[name]
        dist_before = trials_before[0].distributions[name]
        dist_after = trials_after[0].distributions[name]
        assert isinstance(dist_before, (IntDistribution, FloatDistribution))
        assert isinstance(dist_after, (IntDistribution, FloatDistribution))
        assert np.log(dist_before.low) == dist_after.low
        assert np.log(dist_before.high) == dist_after.high

@pytest.mark.parametrize('direction', (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
@pytest.mark.parametrize('top_trials_ratio,min_n_trials,n_trials_expected', [(0.4, 3, 3), (0.8, 3, 4)])
def test_select_top_trials(direction: StudyDirection, top_trials_ratio: float, min_n_trials: int, n_trials_expected: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    values = [1, 0, 0, 2, 1]
    trials_before = [create_trial(value=v) for v in values]
    values_in_order = sorted(values, reverse=direction == StudyDirection.MAXIMIZE)
    p = _preprocessing.SelectTopTrials(top_trials_ratio=top_trials_ratio, min_n_trials=min_n_trials)
    trials_after = p.apply(trials_before, direction)
    assert len(trials_after) == n_trials_expected
    assert _get_trial_values(trials_after) == values_in_order[:n_trials_expected]

@pytest.mark.parametrize('direction', (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_to_minimize(direction: StudyDirection) -> None:
    if False:
        i = 10
        return i + 15
    values = [1, 0, 0, 2, 1]
    trials_before = [create_trial(value=v) for v in values]
    expected_values = values[:]
    if direction == StudyDirection.MAXIMIZE:
        expected_values = [-1 * v for v in expected_values]
    p = _preprocessing.ToMinimize()
    trials_after = p.apply(trials_before, direction)
    assert _get_trial_values(trials_after) == expected_values

@pytest.mark.parametrize('direction', (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_one_to_hot(direction: StudyDirection) -> None:
    if False:
        i = 10
        return i + 15
    trials_before = [create_trial(params={'categorical': 0}, distributions={'categorical': CategoricalDistribution((0, 1))}, value=1.0), create_trial(params={'categorical': 1}, distributions={'categorical': CategoricalDistribution((0, 1))}, value=1.0)]
    p = _preprocessing.OneToHot()
    trials_after = p.apply(trials_before, direction)
    for t in trials_after:
        assert t.distributions == {'i0_categorical': FloatDistribution(0, 1), 'i1_categorical': FloatDistribution(0, 1)}
    assert len(trials_after) == 2
    assert trials_after[0].params == {'i0_categorical': 1.0, 'i1_categorical': 0.0}
    assert trials_after[1].params == {'i0_categorical': 0.0, 'i1_categorical': 1.0}
    trials_before = [create_trial(params={'float': 1, 'int': 3}, distributions={'float': FloatDistribution(1, 10), 'int': IntDistribution(1, 10)}, value=1.0)]
    p = _preprocessing.OneToHot()
    trials_after = p.apply(trials_before, direction)
    assert len(trials_after) == 1
    assert trials_after[0].params == {'i0_float': 1, 'i0_int': 3}
    assert trials_after[0].distributions == {'i0_float': FloatDistribution(1, 10), 'i0_int': IntDistribution(1, 10)}

@pytest.mark.parametrize('direction', (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_add_random_inputs(direction: StudyDirection) -> None:
    if False:
        while True:
            i = 10
    n_additional_trials = 3
    dummy_value = -1
    distributions = {'bacon': CategoricalDistribution((0, 1, 2)), 'egg': FloatDistribution(1, 10), 'spam': IntDistribution(1, 10)}
    trials_before = [create_trial(params={'bacon': 0, 'egg': 1, 'spam': 10}, distributions=distributions, value=1.0)]
    p = _preprocessing.AddRandomInputs(n_additional_trials=n_additional_trials, dummy_value=dummy_value)
    trials_after = p.apply(trials_before, direction)
    assert len(trials_after) == len(trials_before) + n_additional_trials
    assert trials_before[0] == trials_after[0]
    for t in trials_after[1:]:
        assert t.value == dummy_value
        assert t.distributions == distributions
        assert set(t.params.keys()) == set(distributions.keys())
        for (name, distribution) in distributions.items():
            assert distribution._contains(t.params[name])