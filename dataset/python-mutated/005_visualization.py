"""
.. _visualization:

Quick Visualization for Hyperparameter Optimization Analysis
============================================================

Optuna provides various visualization features in :mod:`optuna.visualization` to analyze optimization results visually.

This tutorial walks you through this module by visualizing the history of lightgbm model for breast cancer dataset.

For visualizing multi-objective optimization (i.e., the usage of :func:`optuna.visualization.plot_pareto_front`),
please refer to the tutorial of :ref:`multi_objective`.

.. note::
   By using `Optuna Dashboard <https://github.com/optuna/optuna-dashboard>`_, you can also check the optimization history,
   hyperparameter importances, hyperparameter relationships, etc. in graphs and tables.
   Please make your study persistent using :ref:`RDB backend <rdb>` and execute following commands to run Optuna Dashboard.

   .. code-block:: console

      $ pip install optuna-dashboard
      $ optuna-dashboard sqlite:///example-study.db

   Please check out `the GitHub repository <https://github.com/optuna/optuna-dashboard>`_ for more details.

   .. list-table::
      :header-rows: 1

      * - Manage Studies
        - Visualize with Interactive Graphs
      * - .. image:: https://user-images.githubusercontent.com/5564044/205545958-305f2354-c7cd-4687-be2f-9e46e7401838.gif
        - .. image:: https://user-images.githubusercontent.com/5564044/205545965-278cd7f4-da7d-4e2e-ac31-6d81b106cada.gif
"""
import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline
SEED = 42
np.random.seed(SEED)

def objective(trial):
    if False:
        i = 10
        return i + 15
    (data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
    (train_x, valid_x, train_y, valid_y) = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dvalid = lgb.Dataset(valid_x, label=valid_y)
    param = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt', 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0), 'bagging_freq': trial.suggest_int('bagging_freq', 1, 7), 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)}
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'auc')
    gbm = lgb.train(param, dtrain, valid_sets=[dvalid], callbacks=[pruning_callback])
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED), pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
study.optimize(objective, n_trials=100, timeout=600)
plot_optimization_history(study)
plot_intermediate_values(study)
plot_parallel_coordinate(study)
plot_parallel_coordinate(study, params=['bagging_freq', 'bagging_fraction'])
plot_contour(study)
plot_contour(study, params=['bagging_freq', 'bagging_fraction'])
plot_slice(study)
plot_slice(study, params=['bagging_freq', 'bagging_fraction'])
plot_param_importances(study)
optuna.visualization.plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name='duration')
plot_edf(study)
plot_rank(study)
plot_timeline(study)