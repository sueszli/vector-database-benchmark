from collections import Counter
from typing import List
from typing import Tuple
from unittest.mock import patch
import pytest
import optuna
from optuna import multi_objective
from optuna.study._study_direction import StudyDirection
pytestmark = pytest.mark.filterwarnings('ignore::FutureWarning')

def test_population_size() -> None:
    if False:
        return 10
    sampler = multi_objective.samplers.NSGAIIMultiObjectiveSampler(population_size=10)
    study = multi_objective.create_study(['minimize'], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float('x', 0, 9)], n_trials=40)
    generations = Counter([t.system_attrs[multi_objective.samplers._nsga2._GENERATION_KEY] for t in study.trials])
    assert generations == {0: 10, 1: 10, 2: 10, 3: 10}
    sampler = multi_objective.samplers.NSGAIIMultiObjectiveSampler(population_size=2)
    study = multi_objective.create_study(['minimize'], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float('x', 0, 9)], n_trials=40)
    generations = Counter([t.system_attrs[multi_objective.samplers._nsga2._GENERATION_KEY] for t in study.trials])
    assert generations == {i: 2 for i in range(20)}
    with pytest.raises(ValueError):
        multi_objective.samplers.NSGAIIMultiObjectiveSampler(population_size=1)
    with pytest.raises(TypeError):
        multi_objective.samplers.NSGAIIMultiObjectiveSampler(population_size=2.5)

def test_mutation_prob() -> None:
    if False:
        return 10
    multi_objective.samplers.NSGAIIMultiObjectiveSampler(mutation_prob=None)
    multi_objective.samplers.NSGAIIMultiObjectiveSampler(mutation_prob=0.0)
    multi_objective.samplers.NSGAIIMultiObjectiveSampler(mutation_prob=0.5)
    multi_objective.samplers.NSGAIIMultiObjectiveSampler(mutation_prob=1.0)
    with pytest.raises(ValueError):
        multi_objective.samplers.NSGAIIMultiObjectiveSampler(mutation_prob=-0.5)
    with pytest.raises(ValueError):
        multi_objective.samplers.NSGAIIMultiObjectiveSampler(mutation_prob=1.1)

def test_crossover_prob() -> None:
    if False:
        print('Hello World!')
    multi_objective.samplers.NSGAIIMultiObjectiveSampler(crossover_prob=0.0)
    multi_objective.samplers.NSGAIIMultiObjectiveSampler(crossover_prob=0.5)
    multi_objective.samplers.NSGAIIMultiObjectiveSampler(crossover_prob=1.0)
    with pytest.raises(ValueError):
        multi_objective.samplers.NSGAIIMultiObjectiveSampler(crossover_prob=-0.5)
    with pytest.raises(ValueError):
        multi_objective.samplers.NSGAIIMultiObjectiveSampler(crossover_prob=1.1)

def test_swapping_prob() -> None:
    if False:
        return 10
    multi_objective.samplers.NSGAIIMultiObjectiveSampler(swapping_prob=0.0)
    multi_objective.samplers.NSGAIIMultiObjectiveSampler(swapping_prob=0.5)
    multi_objective.samplers.NSGAIIMultiObjectiveSampler(swapping_prob=1.0)
    with pytest.raises(ValueError):
        multi_objective.samplers.NSGAIIMultiObjectiveSampler(swapping_prob=-0.5)
    with pytest.raises(ValueError):
        multi_objective.samplers.NSGAIIMultiObjectiveSampler(swapping_prob=1.1)

def test_fast_non_dominated_sort() -> None:
    if False:
        return 10
    directions = [StudyDirection.MINIMIZE]
    trials = [_create_frozen_trial(0, [10]), _create_frozen_trial(1, [20]), _create_frozen_trial(2, [20]), _create_frozen_trial(3, [30])]
    population_per_rank = multi_objective.samplers._nsga2._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [{0}, {1, 2}, {3}]
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MAXIMIZE]
    trials = [_create_frozen_trial(0, [10, 30]), _create_frozen_trial(1, [10, 10]), _create_frozen_trial(2, [20, 20]), _create_frozen_trial(3, [30, 10]), _create_frozen_trial(4, [15, 15])]
    population_per_rank = multi_objective.samplers._nsga2._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [{0, 2, 3}, {4}, {1}]
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
    trials = [_create_frozen_trial(0, [5, 5, 4]), _create_frozen_trial(1, [5, 5, 5]), _create_frozen_trial(2, [9, 9, 0]), _create_frozen_trial(3, [5, 7, 5]), _create_frozen_trial(4, [0, 0, 9]), _create_frozen_trial(5, [0, 9, 9])]
    population_per_rank = multi_objective.samplers._nsga2._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [{2}, {0, 3, 5}, {1}, {4}]

def test_crowding_distance_sort() -> None:
    if False:
        while True:
            i = 10
    trials = [_create_frozen_trial(0, [5]), _create_frozen_trial(1, [6]), _create_frozen_trial(2, [9]), _create_frozen_trial(3, [0])]
    multi_objective.samplers._nsga2._crowding_distance_sort(trials)
    assert [t.number for t in trials] == [2, 3, 0, 1]
    trials = [_create_frozen_trial(0, [5, 0]), _create_frozen_trial(1, [6, 0]), _create_frozen_trial(2, [9, 0]), _create_frozen_trial(3, [0, 0])]
    multi_objective.samplers._nsga2._crowding_distance_sort(trials)
    assert [t.number for t in trials] == [2, 3, 0, 1]

def test_study_system_attr_for_population_cache() -> None:
    if False:
        print('Hello World!')
    sampler = multi_objective.samplers.NSGAIIMultiObjectiveSampler(population_size=10)
    study = multi_objective.create_study(['minimize'], sampler=sampler)

    def get_cached_entries(study: multi_objective.study.MultiObjectiveStudy) -> List[Tuple[int, List[int]]]:
        if False:
            return 10
        study_system_attrs = study._storage.get_study_system_attrs(study._study_id)
        return [v for (k, v) in study_system_attrs.items() if k.startswith(multi_objective.samplers._nsga2._POPULATION_CACHE_KEY_PREFIX)]
    study.optimize(lambda t: [t.suggest_float('x', 0, 9)], n_trials=10)
    cached_entries = get_cached_entries(study)
    assert len(cached_entries) == 0
    study.optimize(lambda t: [t.suggest_float('x', 0, 9)], n_trials=1)
    cached_entries = get_cached_entries(study)
    assert len(cached_entries) == 1
    assert cached_entries[0][0] == 0
    assert len(cached_entries[0][1]) == 10
    study.optimize(lambda t: [t.suggest_float('x', 0, 9)], n_trials=10)
    cached_entries = get_cached_entries(study)
    assert len(cached_entries) == 1
    assert cached_entries[0][0] == 1
    assert len(cached_entries[0][1]) == 10

def test_reseed_rng() -> None:
    if False:
        while True:
            i = 10
    sampler = multi_objective.samplers.NSGAIIMultiObjectiveSampler(population_size=10)
    original_random_state = sampler._rng.rng.get_state()
    original_random_sampler_random_state = sampler._random_sampler._sampler._rng.rng.get_state()
    with patch.object(sampler._random_sampler, 'reseed_rng', wraps=sampler._random_sampler.reseed_rng) as mock_object:
        sampler.reseed_rng()
        assert mock_object.call_count == 1
    assert str(original_random_state) != str(sampler._rng.rng.get_state())
    assert str(original_random_sampler_random_state) != str(sampler._random_sampler._sampler._rng.rng.get_state())

def _create_frozen_trial(number: int, values: List[float]) -> multi_objective.trial.FrozenMultiObjectiveTrial:
    if False:
        for i in range(10):
            print('nop')
    trial = optuna.trial.FrozenTrial(number=number, trial_id=number, state=optuna.trial.TrialState.COMPLETE, value=None, datetime_start=None, datetime_complete=None, params={}, distributions={}, user_attrs={}, system_attrs={}, intermediate_values=dict(enumerate(values)))
    return multi_objective.trial.FrozenMultiObjectiveTrial(len(values), trial)