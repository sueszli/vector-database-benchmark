"""
.. _first:

Lightweight, versatile, and platform agnostic architecture
==========================================================

Optuna is entirely written in Python and has few dependencies.
This means that we can quickly move to the real example once you get interested in Optuna.


Quadratic Function Example
--------------------------

Usually, Optuna is used to optimize hyperparameters, but as an example,
let's optimize a simple quadratic function: :math:`(x - 2)^2`.
"""
import optuna

def objective(trial):
    if False:
        for i in range(10):
            print('nop')
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2
study = optuna.create_study()
study.optimize(objective, n_trials=100)
best_params = study.best_params
found_x = best_params['x']
print('Found x: {}, (x - 2)^2: {}'.format(found_x, (found_x - 2) ** 2))
study.best_params
study.best_value
study.best_trial
study.trials
for trial in study.trials[:2]:
    print(trial)
len(study.trials)
study.optimize(objective, n_trials=100)
len(study.trials)
best_params = study.best_params
found_x = best_params['x']
print('Found x: {}, (x - 2)^2: {}'.format(found_x, (found_x - 2) ** 2))