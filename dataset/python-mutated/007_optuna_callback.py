"""
.. _optuna_callback:

Callback for Study.optimize
===========================

This tutorial showcases how to use & implement Optuna ``Callback`` for :func:`~optuna.study.Study.optimize`.

``Callback`` is called after every evaluation of ``objective``, and
it takes :class:`~optuna.study.Study` and :class:`~optuna.trial.FrozenTrial` as arguments, and does some work.

:class:`~optuna.integration.MLflowCallback` is a great example.
"""
import optuna

class StopWhenTrialKeepBeingPrunedCallback:

    def __init__(self, threshold: int):
        if False:
            while True:
                i = 10
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if False:
            print('Hello World!')
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0
        if self._consequtive_pruned_count >= self.threshold:
            study.stop()

def objective(trial):
    if False:
        i = 10
        return i + 15
    if trial.number > 4:
        raise optuna.TrialPruned
    return trial.suggest_float('x', 0, 1)
import logging
import sys
optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
study_stop_cb = StopWhenTrialKeepBeingPrunedCallback(2)
study = optuna.create_study()
study.optimize(objective, n_trials=10, callbacks=[study_stop_cb])