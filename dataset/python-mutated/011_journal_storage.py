"""
.. _journal_storage:

(File-based) Journal Storage
============================

Optuna provides :class:`~optuna.storages.JournalStorage`. With this feature, you can easily run a
distributed optimization over network using NFS as the shared storage, without need for setting up
RDB or Redis.

"""
import logging
import sys
import optuna
optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
study_name = 'example-study'
storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage('./journal.log'))
study = optuna.create_study(study_name=study_name, storage=storage)

def objective(trial):
    if False:
        i = 10
        return i + 15
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2
study.optimize(objective, n_trials=3)