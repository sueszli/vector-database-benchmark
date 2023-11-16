import optuna
from aim.optuna import AimCallback
aim_callback = AimCallback(experiment_name='example_experiment_single_run')

def objective(trial):
    if False:
        i = 10
        return i + 15
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2
study = optuna.create_study()
study.optimize(objective, n_trials=10, callbacks=[aim_callback])
aim_callback = AimCallback(as_multirun=True, experiment_name='example_experiment_multy_run_with_decorator')

@aim_callback.track_in_aim()
def objective(trial):
    if False:
        while True:
            i = 10
    x = trial.suggest_float('x', -10, 10)
    aim_callback.experiment.track_auto(2, name='power')
    aim_callback.experiment.track_auto(x - 2, name='base of metric')
    return (x - 2) ** 2
study = optuna.create_study()
study.optimize(objective, n_trials=10, callbacks=[aim_callback])