from typing import Tuple
import pytest
import optuna

@pytest.mark.parametrize('direction_value', [('minimize', 2), ('maximize', 0.5)])
def test_successive_halving_pruner_intermediate_values(direction_value: Tuple[str, float]) -> None:
    if False:
        for i in range(10):
            print('nop')
    (direction, intermediate_value) = direction_value
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.study.create_study(direction=direction, pruner=pruner)
    trial = study.ask()
    trial.report(1, 1)
    assert not trial.should_prune()
    trial = study.ask()
    assert not trial.should_prune()
    trial.report(intermediate_value, 1)
    assert trial.should_prune()

def test_successive_halving_pruner_rung_check() -> None:
    if False:
        for i in range(10):
            print('nop')
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.study.create_study(pruner=pruner)
    for i in range(7):
        trial = study.ask()
        trial.report(0.1 * (i + 1), step=7)
        pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))
    trial = study.ask()
    trial.report(0.75, step=7)
    pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_0' in trial_system_attrs
    assert 'completed_rung_1' not in trial_system_attrs
    trial = study.ask()
    trial.report(0.25, step=7)
    trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_1' in trial_system_attrs
    assert 'completed_rung_2' not in trial_system_attrs
    trial = study.ask()
    trial.report(0.05, step=7)
    trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_2' in trial_system_attrs
    assert 'completed_rung_3' not in trial_system_attrs

def test_successive_halving_pruner_first_trial_is_not_pruned() -> None:
    if False:
        while True:
            i = 10
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    for i in range(10):
        trial.report(1, step=i)
        assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_0' in trial_system_attrs
    assert 'completed_rung_1' in trial_system_attrs
    assert 'completed_rung_2' in trial_system_attrs
    assert 'completed_rung_3' in trial_system_attrs
    assert 'completed_rung_4' not in trial_system_attrs

def test_successive_halving_pruner_with_nan() -> None:
    if False:
        for i in range(10):
            print('nop')
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.study.create_study(pruner=pruner)
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(float('nan'), step=1)
    assert not trial.should_prune()
    trial.report(float('nan'), step=2)
    assert trial.should_prune()

@pytest.mark.parametrize('n_reports', range(3))
@pytest.mark.parametrize('n_trials', [1, 2])
def test_successive_halving_pruner_with_auto_min_resource(n_reports: int, n_trials: int) -> None:
    if False:
        while True:
            i = 10
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource='auto')
    study = optuna.study.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)
    assert pruner._min_resource is None

    def objective(trial: optuna.trial.Trial) -> float:
        if False:
            print('Hello World!')
        for i in range(n_reports):
            trial.report(1.0 / (i + 1), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return 1.0
    study.optimize(objective, n_trials=n_trials)
    if n_reports > 0 and n_trials > 1:
        assert pruner._min_resource is not None and pruner._min_resource > 0
    else:
        assert pruner._min_resource is None

def test_successive_halving_pruner_with_invalid_str_to_min_resource() -> None:
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        optuna.pruners.SuccessiveHalvingPruner(min_resource='fixed')

def test_successive_halving_pruner_min_resource_parameter() -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        optuna.pruners.SuccessiveHalvingPruner(min_resource=0, reduction_factor=2, min_early_stopping_rate=0)
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, step=1)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_0' in trial_system_attrs
    assert 'completed_rung_1' not in trial_system_attrs
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, step=1)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_0' not in trial_system_attrs
    trial.report(1, step=2)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_0' in trial_system_attrs
    assert 'completed_rung_1' not in trial_system_attrs

def test_successive_halving_pruner_reduction_factor_parameter() -> None:
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):
        optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=1, min_early_stopping_rate=0)
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, step=1)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_0' in trial_system_attrs
    assert 'completed_rung_1' not in trial_system_attrs
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3, min_early_stopping_rate=0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, step=1)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_0' in trial_system_attrs
    assert 'completed_rung_1' not in trial_system_attrs
    trial.report(1, step=2)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_1' not in trial_system_attrs
    trial.report(1, step=3)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_1' in trial_system_attrs
    assert 'completed_rung_2' not in trial_system_attrs

def test_successive_halving_pruner_min_early_stopping_rate_parameter() -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=-1)
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, step=1)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_0' in trial_system_attrs
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=1)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, step=1)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_0' not in trial_system_attrs
    assert 'completed_rung_1' not in trial_system_attrs
    trial.report(1, step=2)
    assert not trial.should_prune()
    trial_system_attrs = trial.storage.get_trial_system_attrs(trial._trial_id)
    assert 'completed_rung_0' in trial_system_attrs
    assert 'completed_rung_1' not in trial_system_attrs

def test_successive_halving_pruner_bootstrap_parameter() -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        optuna.pruners.SuccessiveHalvingPruner(bootstrap_count=-1)
    with pytest.raises(ValueError):
        optuna.pruners.SuccessiveHalvingPruner(bootstrap_count=1, min_resource='auto')
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, bootstrap_count=1)
    study = optuna.study.create_study(pruner=pruner)
    trial1 = study.ask()
    trial2 = study.ask()
    trial1.report(1, step=1)
    assert trial1.should_prune()
    trial2.report(1, step=1)
    assert not trial2.should_prune()