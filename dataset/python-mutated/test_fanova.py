from typing import Tuple
import pytest
from optuna import create_study
from optuna import Trial
from optuna.distributions import FloatDistribution
from optuna.importance import FanovaImportanceEvaluator
from optuna.samplers import RandomSampler
from optuna.trial import create_trial

def objective(trial: Trial) -> float:
    if False:
        i = 10
        return i + 15
    x1 = trial.suggest_float('x1', 0.1, 3)
    x2 = trial.suggest_float('x2', 0.1, 3, log=True)
    x3 = trial.suggest_float('x3', 2, 4, log=True)
    return x1 + x2 * x3

def multi_objective_function(trial: Trial) -> Tuple[float, float]:
    if False:
        return 10
    x1 = trial.suggest_float('x1', 0.1, 3)
    x2 = trial.suggest_float('x2', 0.1, 3, log=True)
    x3 = trial.suggest_float('x3', 2, 4, log=True)
    return (x1, x2 * x3)

def test_fanova_importance_evaluator_n_trees() -> None:
    if False:
        i = 10
        return i + 15
    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)
    evaluator = FanovaImportanceEvaluator(n_trees=10, seed=0)
    param_importance = evaluator.evaluate(study)
    evaluator = FanovaImportanceEvaluator(n_trees=20, seed=0)
    param_importance_different_n_trees = evaluator.evaluate(study)
    assert param_importance != param_importance_different_n_trees

def test_fanova_importance_evaluator_max_depth() -> None:
    if False:
        print('Hello World!')
    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)
    evaluator = FanovaImportanceEvaluator(max_depth=1, seed=0)
    param_importance = evaluator.evaluate(study)
    evaluator = FanovaImportanceEvaluator(max_depth=2, seed=0)
    param_importance_different_max_depth = evaluator.evaluate(study)
    assert param_importance != param_importance_different_max_depth

@pytest.mark.parametrize('inf_value', [float('inf'), -float('inf')])
def test_fanova_importance_evaluator_with_infinite(inf_value: float) -> None:
    if False:
        return 10
    n_trial = 10
    seed = 13
    study = create_study(sampler=RandomSampler(seed=seed))
    study.optimize(objective, n_trials=n_trial)
    evaluator = FanovaImportanceEvaluator(seed=seed)
    param_importance_without_inf = evaluator.evaluate(study)
    study.add_trial(create_trial(value=inf_value, params={'x1': 1.0, 'x2': 1.0, 'x3': 3.0}, distributions={'x1': FloatDistribution(low=0.1, high=3), 'x2': FloatDistribution(low=0.1, high=3, log=True), 'x3': FloatDistribution(low=2, high=4, log=True)}))
    param_importance_with_inf = evaluator.evaluate(study)
    assert param_importance_with_inf == param_importance_without_inf

@pytest.mark.parametrize('target_idx', [0, 1])
@pytest.mark.parametrize('inf_value', [float('inf'), -float('inf')])
def test_multi_objective_fanova_importance_evaluator_with_infinite(target_idx: int, inf_value: float) -> None:
    if False:
        for i in range(10):
            print('nop')
    n_trial = 10
    seed = 13
    study = create_study(directions=['minimize', 'minimize'], sampler=RandomSampler(seed=seed))
    study.optimize(multi_objective_function, n_trials=n_trial)
    evaluator = FanovaImportanceEvaluator(seed=seed)
    param_importance_without_inf = evaluator.evaluate(study, target=lambda t: t.values[target_idx])
    study.add_trial(create_trial(values=[inf_value, inf_value], params={'x1': 1.0, 'x2': 1.0, 'x3': 3.0}, distributions={'x1': FloatDistribution(low=0.1, high=3), 'x2': FloatDistribution(low=0.1, high=3, log=True), 'x3': FloatDistribution(low=2, high=4, log=True)}))
    param_importance_with_inf = evaluator.evaluate(study, target=lambda t: t.values[target_idx])
    assert param_importance_with_inf == param_importance_without_inf