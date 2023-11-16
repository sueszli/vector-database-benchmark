"""
.. _pruning:

Efficient Optimization Algorithms
=================================

Optuna enables efficient hyperparameter optimization by
adopting state-of-the-art algorithms for sampling hyperparameters and
pruning efficiently unpromising trials.

Sampling Algorithms
-------------------

Samplers basically continually narrow down the search space using the records of suggested parameter values and evaluated objective values,
leading to an optimal search space which giving off parameters leading to better objective values.
More detailed explanation of how samplers suggest parameters is in :class:`~optuna.samplers.BaseSampler`.

Optuna provides the following sampling algorithms:

- Grid Search implemented in :class:`~optuna.samplers.GridSampler`

- Random Search implemented in :class:`~optuna.samplers.RandomSampler`

- Tree-structured Parzen Estimator algorithm implemented in :class:`~optuna.samplers.TPESampler`

- CMA-ES based algorithm implemented in :class:`~optuna.samplers.CmaEsSampler`

- Algorithm to enable partial fixed parameters implemented in :class:`~optuna.samplers.PartialFixedSampler`

- Nondominated Sorting Genetic Algorithm II implemented in :class:`~optuna.samplers.NSGAIISampler`

- A Quasi Monte Carlo sampling algorithm implemented in :class:`~optuna.samplers.QMCSampler`

The default sampler is :class:`~optuna.samplers.TPESampler`.

Switching Samplers
------------------

"""
import optuna
study = optuna.create_study()
print(f'Sampler is {study.sampler.__class__.__name__}')
study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
print(f'Sampler is {study.sampler.__class__.__name__}')
study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
print(f'Sampler is {study.sampler.__class__.__name__}')
import logging
import sys
import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

def objective(trial):
    if False:
        return 10
    iris = sklearn.datasets.load_iris()
    classes = list(set(iris.target))
    (train_x, valid_x, train_y, valid_y) = sklearn.model_selection.train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)
    alpha = trial.suggest_float('alpha', 1e-05, 0.1, log=True)
    clf = sklearn.linear_model.SGDClassifier(alpha=alpha)
    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)
        intermediate_value = 1.0 - clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return 1.0 - clf.score(valid_x, valid_y)
optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)