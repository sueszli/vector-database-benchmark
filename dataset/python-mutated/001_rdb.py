"""
.. _rdb:

Saving/Resuming Study with RDB Backend
==========================================

An RDB backend enables persistent experiments (i.e., to save and resume a study) as well as access to history of studies.
In addition, we can run multi-node optimization tasks with this feature, which is described in :ref:`distributed`.

In this section, let's try simple examples running on a local environment with SQLite DB.

.. note::
    You can also utilize other RDB backends, e.g., PostgreSQL or MySQL, by setting the storage argument to the DB's URL.
    Please refer to `SQLAlchemy's document <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_ for how to set up the URL.


New Study
---------

We can create a persistent study by calling :func:`~optuna.study.create_study` function as follows.
An SQLite file ``example.db`` is automatically initialized with a new study record.
"""
import logging
import sys
import optuna
optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
study_name = 'example-study'
storage_name = 'sqlite:///{}.db'.format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name)

def objective(trial):
    if False:
        for i in range(10):
            print('nop')
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2
study.optimize(objective, n_trials=3)
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=3)
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
print(df)
print('Best params: ', study.best_params)
print('Best value: ', study.best_value)
print('Best Trial: ', study.best_trial)
print('Trials: ', study.trials)