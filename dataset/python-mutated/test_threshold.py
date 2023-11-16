import pytest
import optuna

def test_threshold_pruner_with_ub() -> None:
    if False:
        return 10
    pruner = optuna.pruners.ThresholdPruner(upper=2.0, n_warmup_steps=0, interval_steps=1)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1.0, 1)
    assert not trial.should_prune()
    trial.report(3.0, 2)
    assert trial.should_prune()

def test_threshold_pruner_with_lt() -> None:
    if False:
        i = 10
        return i + 15
    pruner = optuna.pruners.ThresholdPruner(lower=2.0, n_warmup_steps=0, interval_steps=1)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(3.0, 1)
    assert not trial.should_prune()
    trial.report(1.0, 2)
    assert trial.should_prune()

def test_threshold_pruner_with_two_side() -> None:
    if False:
        while True:
            i = 10
    pruner = optuna.pruners.ThresholdPruner(lower=0.0, upper=1.0, n_warmup_steps=0, interval_steps=1)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(-0.1, 1)
    assert trial.should_prune()
    trial.report(0.0, 2)
    assert not trial.should_prune()
    trial.report(0.4, 3)
    assert not trial.should_prune()
    trial.report(1.0, 4)
    assert not trial.should_prune()
    trial.report(1.1, 5)
    assert trial.should_prune()

def test_threshold_pruner_with_invalid_inputs() -> None:
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError):
        optuna.pruners.ThresholdPruner(lower='val', upper=1.0)
    with pytest.raises(TypeError):
        optuna.pruners.ThresholdPruner(lower=0.0, upper='val')
    with pytest.raises(TypeError):
        optuna.pruners.ThresholdPruner(lower=None, upper=None)

def test_threshold_pruner_with_nan() -> None:
    if False:
        return 10
    pruner = optuna.pruners.ThresholdPruner(lower=0.0, upper=1.0, n_warmup_steps=0, interval_steps=1)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(float('nan'), 1)
    assert trial.should_prune()

def test_threshold_pruner_n_warmup_steps() -> None:
    if False:
        print('Hello World!')
    pruner = optuna.pruners.ThresholdPruner(lower=0.0, upper=1.0, n_warmup_steps=2)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(-10.0, 0)
    assert not trial.should_prune()
    trial.report(100.0, 1)
    assert not trial.should_prune()
    trial.report(-100.0, 3)
    assert trial.should_prune()
    trial.report(1.0, 4)
    assert not trial.should_prune()
    trial.report(1000.0, 5)
    assert trial.should_prune()

def test_threshold_pruner_interval_steps() -> None:
    if False:
        print('Hello World!')
    pruner = optuna.pruners.ThresholdPruner(lower=0.0, upper=1.0, interval_steps=2)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(-10.0, 0)
    assert trial.should_prune()
    trial.report(100.0, 1)
    assert not trial.should_prune()
    trial.report(-100.0, 2)
    assert trial.should_prune()
    trial.report(10.0, 3)
    assert not trial.should_prune()
    trial.report(1000.0, 4)
    assert trial.should_prune()