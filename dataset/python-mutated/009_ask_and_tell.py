"""
.. _ask_and_tell:

Ask-and-Tell Interface
=======================

Optuna has an `Ask-and-Tell` interface, which provides a more flexible interface for hyperparameter optimization.
This tutorial explains three use-cases when the ask-and-tell interface is beneficial:

- :ref:`Apply-optuna-to-an-existing-optimization-problem-with-minimum-modifications`
- :ref:`Define-and-Run`
- :ref:`Batch-Optimization`

.. _Apply-optuna-to-an-existing-optimization-problem-with-minimum-modifications:

----------------------------------------------------------------------------
Apply Optuna to an existing optimization problem with minimum modifications
----------------------------------------------------------------------------

Let's consider the traditional supervised classification problem; you aim to maximize the validation accuracy.
To do so, you train `LogisticRegression` as a simple model.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import optuna
(X, y) = make_classification(n_features=10)
(X_train, X_test, y_train, y_test) = train_test_split(X, y)
C = 0.01
clf = LogisticRegression(C=C)
clf.fit(X_train, y_train)
val_accuracy = clf.score(X_test, y_test)

def objective(trial):
    if False:
        return 10
    (X, y) = make_classification(n_features=10)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y)
    C = trial.suggest_float('C', 1e-07, 10.0, log=True)
    solver = trial.suggest_categorical('solver', ('lbfgs', 'saga'))
    clf = LogisticRegression(C=C, solver=solver)
    clf.fit(X_train, y_train)
    val_accuracy = clf.score(X_test, y_test)
    return val_accuracy
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
study = optuna.create_study(direction='maximize')
n_trials = 10
for _ in range(n_trials):
    trial = study.ask()
    C = trial.suggest_float('C', 1e-07, 10.0, log=True)
    solver = trial.suggest_categorical('solver', ('lbfgs', 'saga'))
    clf = LogisticRegression(C=C, solver=solver)
    clf.fit(X_train, y_train)
    val_accuracy = clf.score(X_test, y_test)
    study.tell(trial, val_accuracy)
distributions = {'C': optuna.distributions.FloatDistribution(1e-07, 10.0, log=True), 'solver': optuna.distributions.CategoricalDistribution(('lbfgs', 'saga'))}
study = optuna.create_study(direction='maximize')
n_trials = 10
for _ in range(n_trials):
    trial = study.ask(distributions)
    C = trial.params['C']
    solver = trial.params['solver']
    clf = LogisticRegression(C=C, solver=solver)
    clf.fit(X_train, y_train)
    val_accuracy = clf.score(X_test, y_test)
    study.tell(trial, val_accuracy)

def batched_objective(xs: np.ndarray, ys: np.ndarray):
    if False:
        i = 10
        return i + 15
    return xs ** 2 + ys
batch_size = 10
study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
for _ in range(3):
    trial_numbers = []
    x_batch = []
    y_batch = []
    for _ in range(batch_size):
        trial = study.ask()
        trial_numbers.append(trial.number)
        x_batch.append(trial.suggest_float('x', -10, 10))
        y_batch.append(trial.suggest_float('y', -10, 10))
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    objectives = batched_objective(x_batch, y_batch)
    for (trial_number, objective) in zip(trial_numbers, objectives):
        study.tell(trial_number, objective)