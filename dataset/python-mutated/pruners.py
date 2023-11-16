from __future__ import annotations
import optuna

class DeterministicPruner(optuna.pruners.BasePruner):

    def __init__(self, is_pruning: bool) -> None:
        if False:
            return 10
        self.is_pruning = is_pruning

    def prune(self, study: 'optuna.study.Study', trial: 'optuna.trial.FrozenTrial') -> bool:
        if False:
            while True:
                i = 10
        return self.is_pruning