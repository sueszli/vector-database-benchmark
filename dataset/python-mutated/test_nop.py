import optuna

def test_nop_pruner() -> None:
    if False:
        for i in range(10):
            print('nop')
    pruner = optuna.pruners.NopPruner()
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, 1)
    assert not trial.should_prune()