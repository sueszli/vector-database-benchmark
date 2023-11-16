from typing import List
import pytest
import optuna

def test_patient_pruner_experimental_warning() -> None:
    if False:
        return 10
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.pruners.PatientPruner(None, 0)

def test_patient_pruner_patience() -> None:
    if False:
        return 10
    optuna.pruners.PatientPruner(None, 0)
    optuna.pruners.PatientPruner(None, 1)
    with pytest.raises(ValueError):
        optuna.pruners.PatientPruner(None, -1)

def test_patient_pruner_min_delta() -> None:
    if False:
        return 10
    optuna.pruners.PatientPruner(None, 0, 0.0)
    optuna.pruners.PatientPruner(None, 0, 1.0)
    with pytest.raises(ValueError):
        optuna.pruners.PatientPruner(None, 0, -1)

def test_patient_pruner_with_one_trial() -> None:
    if False:
        i = 10
        return i + 15
    pruner = optuna.pruners.PatientPruner(None, 0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, 0)
    assert not trial.should_prune()

@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_patient_pruner_intermediate_values_nan() -> None:
    if False:
        print('Hello World!')
    pruner = optuna.pruners.PatientPruner(None, 0, 0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    assert not trial.should_prune()
    trial.report(float('nan'), 0)
    assert not trial.should_prune()
    trial.report(1.0, 1)
    assert not trial.should_prune()
    trial.report(float('nan'), 2)
    assert not trial.should_prune()

@pytest.mark.parametrize('patience,min_delta,direction,intermediates,expected_prune_steps', [(0, 0, 'maximize', [1, 0], [1]), (1, 0, 'maximize', [2, 1, 0], [2]), (0, 0, 'minimize', [0, 1], [1]), (1, 0, 'minimize', [0, 1, 2], [2]), (0, 1.0, 'maximize', [1, 0], []), (1, 1.0, 'maximize', [3, 2, 1, 0], [3]), (0, 1.0, 'minimize', [0, 1], []), (1, 1.0, 'minimize', [0, 1, 2, 3], [3])])
def test_patient_pruner_intermediate_values(patience: int, min_delta: float, direction: str, intermediates: List[int], expected_prune_steps: List[int]) -> None:
    if False:
        return 10
    pruner = optuna.pruners.PatientPruner(None, patience, min_delta)
    study = optuna.study.create_study(pruner=pruner, direction=direction)
    trial = study.ask()
    pruned = []
    for (step, value) in enumerate(intermediates):
        trial.report(value, step)
        if trial.should_prune():
            pruned.append(step)
    assert pruned == expected_prune_steps