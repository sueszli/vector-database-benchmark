import pytest
from optuna import create_study
from optuna import TrialPruned
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.search_space import _GroupDecomposedSearchSpace
from optuna.search_space import _SearchSpaceGroup
from optuna.testing.storages import StorageSupplier
from optuna.trial import Trial

def test_search_space_group() -> None:
    if False:
        i = 10
        return i + 15
    search_space_group = _SearchSpaceGroup()
    assert search_space_group.search_spaces == []
    search_space_group.add_distributions({})
    assert search_space_group.search_spaces == []
    search_space_group.add_distributions({'x': IntDistribution(low=0, high=10)})
    assert search_space_group.search_spaces == [{'x': IntDistribution(low=0, high=10)}]
    search_space_group.add_distributions({'x': IntDistribution(low=0, high=10)})
    assert search_space_group.search_spaces == [{'x': IntDistribution(low=0, high=10)}]
    search_space_group.add_distributions({'y': IntDistribution(low=0, high=10), 'z': FloatDistribution(low=-3, high=3)})
    assert search_space_group.search_spaces == [{'x': IntDistribution(low=0, high=10)}, {'y': IntDistribution(low=0, high=10), 'z': FloatDistribution(low=-3, high=3)}]
    search_space_group.add_distributions({'y': IntDistribution(low=0, high=10), 'z': FloatDistribution(low=-3, high=3), 'u': FloatDistribution(low=0.01, high=100.0, log=True), 'v': CategoricalDistribution(choices=['A', 'B', 'C'])})
    assert search_space_group.search_spaces == [{'x': IntDistribution(low=0, high=10)}, {'y': IntDistribution(low=0, high=10), 'z': FloatDistribution(low=-3, high=3)}, {'u': FloatDistribution(low=0.01, high=100.0, log=True), 'v': CategoricalDistribution(choices=['A', 'B', 'C'])}]
    search_space_group.add_distributions({'u': FloatDistribution(low=0.01, high=100.0, log=True)})
    assert search_space_group.search_spaces == [{'x': IntDistribution(low=0, high=10)}, {'y': IntDistribution(low=0, high=10), 'z': FloatDistribution(low=-3, high=3)}, {'u': FloatDistribution(low=0.01, high=100.0, log=True)}, {'v': CategoricalDistribution(choices=['A', 'B', 'C'])}]
    search_space_group.add_distributions({'y': IntDistribution(low=0, high=10), 'w': IntDistribution(low=2, high=8, log=True)})
    assert search_space_group.search_spaces == [{'x': IntDistribution(low=0, high=10)}, {'y': IntDistribution(low=0, high=10)}, {'z': FloatDistribution(low=-3, high=3)}, {'u': FloatDistribution(low=0.01, high=100.0, log=True)}, {'v': CategoricalDistribution(choices=['A', 'B', 'C'])}, {'w': IntDistribution(low=2, high=8, log=True)}]
    search_space_group.add_distributions({'y': IntDistribution(low=0, high=10), 'w': IntDistribution(low=2, high=8, log=True), 't': FloatDistribution(low=10, high=100)})
    assert search_space_group.search_spaces == [{'x': IntDistribution(low=0, high=10)}, {'y': IntDistribution(low=0, high=10)}, {'z': FloatDistribution(low=-3, high=3)}, {'u': FloatDistribution(low=0.01, high=100.0, log=True)}, {'v': CategoricalDistribution(choices=['A', 'B', 'C'])}, {'w': IntDistribution(low=2, high=8, log=True)}, {'t': FloatDistribution(low=10, high=100)}]

def test_group_decomposed_search_space() -> None:
    if False:
        return 10
    search_space = _GroupDecomposedSearchSpace()
    study = create_study()
    assert search_space.calculate(study).search_spaces == []
    study.optimize(lambda t: t.suggest_int('x', 0, 10), n_trials=1)
    assert search_space.calculate(study).search_spaces == [{'x': IntDistribution(low=0, high=10)}]
    study.optimize(lambda t: t.suggest_int('y', 0, 10) + t.suggest_float('z', -3, 3), n_trials=1)
    assert search_space.calculate(study).search_spaces == [{'x': IntDistribution(low=0, high=10)}, {'y': IntDistribution(low=0, high=10), 'z': FloatDistribution(low=-3, high=3)}]
    study.optimize(lambda t: t.suggest_int('y', 0, 10) + t.suggest_float('z', -3, 3) + t.suggest_float('u', 0.01, 100.0, log=True) + bool(t.suggest_categorical('v', ['A', 'B', 'C'])), n_trials=1)
    assert search_space.calculate(study).search_spaces == [{'x': IntDistribution(low=0, high=10)}, {'z': FloatDistribution(low=-3, high=3), 'y': IntDistribution(low=0, high=10)}, {'u': FloatDistribution(low=0.01, high=100.0, log=True), 'v': CategoricalDistribution(choices=['A', 'B', 'C'])}]
    study.optimize(lambda t: t.suggest_float('u', 0.01, 100.0, log=True), n_trials=1)
    assert search_space.calculate(study).search_spaces == [{'x': IntDistribution(low=0, high=10)}, {'y': IntDistribution(low=0, high=10), 'z': FloatDistribution(low=-3, high=3)}, {'u': FloatDistribution(low=0.01, high=100.0, log=True)}, {'v': CategoricalDistribution(choices=['A', 'B', 'C'])}]
    study.optimize(lambda t: t.suggest_int('y', 0, 10) + t.suggest_int('w', 2, 8, log=True), n_trials=1)
    assert search_space.calculate(study).search_spaces == [{'x': IntDistribution(low=0, high=10)}, {'y': IntDistribution(low=0, high=10)}, {'z': FloatDistribution(low=-3, high=3)}, {'u': FloatDistribution(low=0.01, high=100.0, log=True)}, {'v': CategoricalDistribution(choices=['A', 'B', 'C'])}, {'w': IntDistribution(low=2, high=8, log=True)}]
    search_space = _GroupDecomposedSearchSpace()
    study = create_study()

    def objective(trial: Trial, exception: Exception) -> float:
        if False:
            while True:
                i = 10
        trial.suggest_float('a', 0, 1)
        raise exception
    study.optimize(lambda t: objective(t, RuntimeError()), n_trials=1, catch=(RuntimeError,))
    study.optimize(lambda t: objective(t, TrialPruned()), n_trials=1)
    assert search_space.calculate(study).search_spaces == []
    study.optimize(lambda t: t.suggest_float('a', -1, 1), n_trials=1)
    study.optimize(lambda t: t.suggest_float('a', 0, 1), n_trials=1)
    assert search_space.calculate(study).search_spaces == [{'a': FloatDistribution(low=-1, high=1)}]

def test_group_decomposed_search_space_with_different_studies() -> None:
    if False:
        for i in range(10):
            print('nop')
    search_space = _GroupDecomposedSearchSpace()
    with StorageSupplier('sqlite') as storage:
        study0 = create_study(storage=storage)
        study1 = create_study(storage=storage)
        search_space.calculate(study0)
        with pytest.raises(ValueError):
            search_space.calculate(study1)