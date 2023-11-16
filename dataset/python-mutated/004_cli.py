"""
.. _cli:

Command-Line Interface
======================

.. csv-table::
   :header: Command, Description
   :widths: 20, 40
   :escape: \\

    ask, Create a new trial and suggest parameters.
    best-trial, Show the best trial.
    best-trials, Show a list of trials located at the Pareto front.
    create-study, Create a new study.
    delete-study, Delete a specified study.
    storage upgrade, Upgrade the schema of a storage.
    studies, Show a list of studies.
    study optimize, Start optimization of a study.
    study set-user-attr, Set a user attribute to a study.
    tell, Finish a trial\\, which was created by the ask command.
    trials, Show a list of trials.

Optuna provides command-line interface as shown in the above table.

Let us assume you are not in IPython shell and writing Python script files instead.
It is totally fine to write scripts like the following:
"""
import optuna

def objective(trial):
    if False:
        i = 10
        return i + 15
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2
if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))