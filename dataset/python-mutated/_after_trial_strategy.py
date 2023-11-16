from __future__ import annotations
from collections.abc import Callable
from collections.abc import Sequence
from optuna.samplers._base import _process_constraints_after_trial
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

class NSGAIIAfterTrialStrategy:

    def __init__(self, *, constraints_func: Callable[[FrozenTrial], Sequence[float]] | None=None) -> None:
        if False:
            return 10
        self._constraints_func = constraints_func

    def __call__(self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float] | None=None) -> None:
        if False:
            print('Hello World!')
        'Carry out the after trial process of default NSGA-II.\n\n        This method is called after each trial of the study, examines whether the trial result is\n        valid in terms of constraints, and store the results in system_attrs of the study.\n        '
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)