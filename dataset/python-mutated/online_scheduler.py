import numpy as np
import logging
from typing import Dict
from flaml.tune.scheduler import TrialScheduler
from flaml.tune import Trial
logger = logging.getLogger(__name__)

class OnlineScheduler(TrialScheduler):
    """Class for the most basic OnlineScheduler."""

    def on_trial_result(self, trial_runner, trial: Trial, result: Dict):
        if False:
            i = 10
            return i + 15
        "Report result and return a decision on the trial's status."
        return TrialScheduler.CONTINUE

    def choose_trial_to_run(self, trial_runner) -> Trial:
        if False:
            while True:
                i = 10
        'Decide which trial to run next.'
        for trial in trial_runner.get_trials():
            if trial.status == Trial.PENDING:
                return trial
        min_paused_resource = np.inf
        min_paused_resource_trial = None
        for trial in trial_runner.get_trials():
            if trial.status == Trial.PAUSED and trial.resource_lease < min_paused_resource:
                min_paused_resource = trial.resource_lease
                min_paused_resource_trial = trial
        if min_paused_resource_trial is not None:
            return min_paused_resource_trial

class OnlineSuccessiveDoublingScheduler(OnlineScheduler):
    """class for the OnlineSuccessiveDoublingScheduler algorithm."""

    def __init__(self, increase_factor: float=2.0):
        if False:
            print('Hello World!')
        'Constructor.\n\n        Args:\n            increase_factor: A float of multiplicative factor\n                used to increase resource lease. Default is 2.0.\n        '
        super().__init__()
        self._increase_factor = increase_factor

    def on_trial_result(self, trial_runner, trial: Trial, result: Dict):
        if False:
            i = 10
            return i + 15
        "Report result and return a decision on the trial's status."
        if trial.result is None or trial.result.resource_used < trial.resource_lease:
            return TrialScheduler.CONTINUE
        else:
            trial.set_resource_lease(trial.resource_lease * self._increase_factor)
            logger.info('Doubled resource for trial %s, used: %s, current budget %s', trial.trial_id, trial.result.resource_used, trial.resource_lease)
            return TrialScheduler.PAUSE

class ChaChaScheduler(OnlineSuccessiveDoublingScheduler):
    """class for the ChaChaScheduler algorithm."""

    def __init__(self, increase_factor: float=2.0, **kwargs):
        if False:
            while True:
                i = 10
        'Constructor.\n\n        Args:\n            increase_factor: A float of multiplicative factor\n                used to increase resource lease. Default is 2.0.\n        '
        super().__init__(increase_factor)
        self._keep_champion = kwargs.get('keep_champion', True)
        self._keep_challenger_metric = kwargs.get('keep_challenger_metric', 'ucb')
        self._keep_challenger_ratio = kwargs.get('keep_challenger_ratio', 0.5)
        self._pause_old_froniter = kwargs.get('pause_old_froniter', False)
        logger.info('Using chacha scheduler with config %s', kwargs)

    def on_trial_result(self, trial_runner, trial: Trial, result: Dict):
        if False:
            print('Hello World!')
        "Report result and return a decision on the trial's status."
        decision = super().on_trial_result(trial_runner, trial, result)
        if self._pause_old_froniter and (not trial.is_checked_under_current_champion):
            if decision == TrialScheduler.CONTINUE:
                decision = TrialScheduler.PAUSE
                trial.set_checked_under_current_champion(True)
                logger.info('Tentitively set trial as paused')
        if self._keep_champion and trial.trial_id == trial_runner.champion_trial.trial_id and (decision == TrialScheduler.PAUSE):
            return TrialScheduler.CONTINUE
        if self._keep_challenger_ratio is not None:
            if decision == TrialScheduler.PAUSE:
                logger.debug('champion, %s', trial_runner.champion_trial.trial_id)
                top_trials = trial_runner.get_top_running_trials(self._keep_challenger_ratio, self._keep_challenger_metric)
                logger.debug('top_learners: %s', top_trials)
                if trial in top_trials:
                    logger.debug('top runner %s: set from PAUSE to CONTINUE', trial.trial_id)
                    return TrialScheduler.CONTINUE
        return decision