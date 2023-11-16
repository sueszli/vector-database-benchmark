"""
.. _configurations:

Pythonic Search Space
=====================

For hyperparameter sampling, Optuna provides the following features:

- :func:`optuna.trial.Trial.suggest_categorical` for categorical parameters
- :func:`optuna.trial.Trial.suggest_int` for integer parameters
- :func:`optuna.trial.Trial.suggest_float` for floating point parameters

With optional arguments of ``step`` and ``log``, we can discretize or take the logarithm of
integer and floating point parameters.
"""
import optuna

def objective(trial):
    if False:
        return 10
    optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    num_channels = trial.suggest_int('num_channels', 32, 512, log=True)
    num_units = trial.suggest_int('num_units', 10, 100, step=5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 1.0)
    learning_rate = trial.suggest_float('learning_rate', 1e-05, 0.01, log=True)
    drop_path_rate = trial.suggest_float('drop_path_rate', 0.0, 1.0, step=0.1)
import sklearn.ensemble
import sklearn.svm

def objective(trial):
    if False:
        return 10
    classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest'])
    if classifier_name == 'SVC':
        svc_c = trial.suggest_float('svc_c', 1e-10, 10000000000.0, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)