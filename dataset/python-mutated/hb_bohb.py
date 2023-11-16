import logging
from typing import Dict, Optional, TYPE_CHECKING
from ray.tune.schedulers.trial_scheduler import TrialScheduler
from ray.tune.schedulers.hyperband import HyperBandScheduler
from ray.tune.experiment import Trial
from ray.util import PublicAPI
if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController
logger = logging.getLogger(__name__)

@PublicAPI
class HyperBandForBOHB(HyperBandScheduler):
    """Extends HyperBand early stopping algorithm for BOHB.

    This implementation removes the ``HyperBandScheduler`` pipelining. This
    class introduces key changes:

    1. Trials are now placed so that the bracket with the largest size is
    filled first.

    2. Trials will be paused even if the bracket is not filled. This allows
    BOHB to insert new trials into the training.

    See ray.tune.schedulers.HyperBandScheduler for parameter docstring.
    """

    def on_trial_add(self, tune_controller: 'TuneController', trial: Trial):
        if False:
            while True:
                i = 10
        'Adds new trial.\n\n        On a new trial add, if current bracket is not filled, add to current\n        bracket. Else, if current band is not filled, create new bracket, add\n        to current bracket. Else, create new iteration, create new bracket,\n        add to bracket.\n        '
        if not self._metric or not self._metric_op:
            raise ValueError('{} has been instantiated without a valid `metric` ({}) or `mode` ({}) parameter. Either pass these parameters when instantiating the scheduler, or pass them as parameters to `tune.TuneConfig()`'.format(self.__class__.__name__, self._metric, self._mode))
        cur_bracket = self._state['bracket']
        cur_band = self._hyperbands[self._state['band_idx']]
        if cur_bracket is None or cur_bracket.filled():
            retry = True
            while retry:
                if self._cur_band_filled():
                    cur_band = []
                    self._hyperbands.append(cur_band)
                    self._state['band_idx'] += 1
                s = self._s_max_1 - len(cur_band) - 1
                assert s >= 0, 'Current band is filled!'
                if self._get_r0(s) == 0:
                    logger.debug('BOHB: Bracket too small - Retrying...')
                    cur_bracket = None
                else:
                    retry = False
                    cur_bracket = self._create_bracket(s)
                cur_band.append(cur_bracket)
                self._state['bracket'] = cur_bracket
        self._state['bracket'].add_trial(trial)
        self._trial_info[trial] = (cur_bracket, self._state['band_idx'])

    def on_trial_result(self, tune_controller: 'TuneController', trial: Trial, result: Dict) -> str:
        if False:
            while True:
                i = 10
        'If bracket is finished, all trials will be stopped.\n\n        If a given trial finishes and bracket iteration is not done,\n        the trial will be paused and resources will be given up.\n\n        This scheduler will not start trials but will stop trials.\n        The current running trial will not be handled,\n        as the trialrunner will be given control to handle it.'
        result['hyperband_info'] = {}
        (bracket, _) = self._trial_info[trial]
        bracket.update_trial_stats(trial, result)
        if bracket.continue_trial(trial):
            return TrialScheduler.CONTINUE
        result['hyperband_info']['budget'] = bracket._cumul_r
        statuses = [(t, t.status) for t in bracket._live_trials]
        if not bracket.filled() or any((status != Trial.PAUSED for (t, status) in statuses if t is not trial)):
            tune_controller.search_alg.searcher.on_pause(trial.trial_id)
            return TrialScheduler.PAUSE
        logger.debug(f'Processing bracket after trial {trial} result')
        action = self._process_bracket(tune_controller, bracket)
        if action == TrialScheduler.PAUSE:
            tune_controller.search_alg.searcher.on_pause(trial.trial_id)
        return action

    def _unpause_trial(self, tune_controller: 'TuneController', trial: Trial):
        if False:
            while True:
                i = 10
        tune_controller.search_alg.searcher.on_unpause(trial.trial_id)

    def choose_trial_to_run(self, tune_controller: 'TuneController', allow_recurse: bool=True) -> Optional[Trial]:
        if False:
            i = 10
            return i + 15
        'Fair scheduling within iteration by completion percentage.\n\n        List of trials not used since all trials are tracked as state\n        of scheduler. If iteration is occupied (ie, no trials to run),\n        then look into next iteration.\n        '
        for hyperband in self._hyperbands:
            scrubbed = [b for b in hyperband if b is not None]
            for bracket in scrubbed:
                for trial in bracket.current_trials():
                    if trial.status == Trial.PAUSED and trial in bracket.trials_to_unpause or trial.status == Trial.PENDING:
                        return trial
        if not any((t.status == Trial.RUNNING for t in tune_controller.get_trials())):
            for hyperband in self._hyperbands:
                for bracket in hyperband:
                    if bracket and any((trial.status == Trial.PAUSED for trial in bracket.current_trials())):
                        logger.debug('Processing bracket since no trial is running.')
                        self._process_bracket(tune_controller, bracket)
                        if allow_recurse and any((trial.status == Trial.PAUSED and trial in bracket.trials_to_unpause or trial.status == Trial.PENDING for trial in bracket.current_trials())):
                            return self.choose_trial_to_run(tune_controller, allow_recurse=False)
        return None