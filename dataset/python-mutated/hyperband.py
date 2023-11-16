import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController
logger = logging.getLogger(__name__)

@PublicAPI
class HyperBandScheduler(FIFOScheduler):
    """Implements the HyperBand early stopping algorithm.

    HyperBandScheduler early stops trials using the HyperBand optimization
    algorithm. It divides trials into brackets of varying sizes, and
    periodically early stops low-performing trials within each bracket.

    To use this implementation of HyperBand with Tune, all you need
    to do is specify the max length of time a trial can run `max_t`, the time
    units `time_attr`, the name of the reported objective value `metric`,
    and if `metric` is to be maximized or minimized (`mode`).
    We automatically determine reasonable values for the other
    HyperBand parameters based on the given values.

    For example, to limit trials to 10 minutes and early stop based on the
    `episode_mean_reward` attr, construct:

    ``HyperBand('time_total_s', 'episode_reward_mean', max_t=600)``

    Note that Tune's stopping criteria will be applied in conjunction with
    HyperBand's early stopping mechanisms.

    See also: https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/

    Args:
        time_attr: The training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        metric: The training result objective value attribute. Stopping
            procedures will use this attribute. If None but a mode was passed,
            the `ray.tune.result.DEFAULT_METRIC` will be used per default.
        mode: One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        max_t: max time units per trial. Trials will be stopped after
            max_t time units (determined by time_attr) have passed.
            The scheduler will terminate trials after this time has passed.
            Note that this is different from the semantics of `max_t` as
            mentioned in the original HyperBand paper.
        reduction_factor: Same as `eta`. Determines how sharp
            the difference is between bracket space-time allocation ratios.
        stop_last_trials: Whether to terminate the trials after
            reaching max_t. Defaults to True.
    """
    _supports_buffered_results = False

    def __init__(self, time_attr: str='training_iteration', metric: Optional[str]=None, mode: Optional[str]=None, max_t: int=81, reduction_factor: float=3, stop_last_trials: bool=True):
        if False:
            while True:
                i = 10
        assert max_t > 0, 'Max (time_attr) not valid!'
        if mode:
            assert mode in ['min', 'max'], "`mode` must be 'min' or 'max'!"
        super().__init__()
        self._eta = reduction_factor
        self._s_max_1 = int(np.round(np.log(max_t) / np.log(reduction_factor))) + 1
        self._max_t_attr = max_t
        self._get_n0 = lambda s: int(np.ceil(self._s_max_1 / (s + 1) * self._eta ** s))
        self._get_r0 = lambda s: int(max_t * self._eta ** (-s))
        self._hyperbands = [[]]
        self._trial_info = {}
        self._state = {'bracket': None, 'band_idx': 0}
        self._num_stopped = 0
        self._metric = metric
        self._mode = mode
        self._metric_op = None
        if self._mode == 'max':
            self._metric_op = 1.0
        elif self._mode == 'min':
            self._metric_op = -1.0
        self._time_attr = time_attr
        self._stop_last_trials = stop_last_trials

    def set_search_properties(self, metric: Optional[str], mode: Optional[str], **spec) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if self._metric and metric:
            return False
        if self._mode and mode:
            return False
        if metric:
            self._metric = metric
        if mode:
            self._mode = mode
        if self._mode == 'max':
            self._metric_op = 1.0
        elif self._mode == 'min':
            self._metric_op = -1.0
        if self._metric is None and self._mode:
            self._metric = DEFAULT_METRIC
        return True

    def on_trial_add(self, tune_controller: 'TuneController', trial: Trial):
        if False:
            while True:
                i = 10
        'Adds new trial.\n\n        On a new trial add, if current bracket is not filled,\n        add to current bracket. Else, if current band is not filled,\n        create new bracket, add to current bracket.\n        Else, create new iteration, create new bracket, add to bracket.'
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
                s = len(cur_band)
                assert s < self._s_max_1, 'Current band is filled!'
                if self._get_r0(s) == 0:
                    logger.info('Bracket too small - Retrying...')
                    cur_bracket = None
                else:
                    retry = False
                    cur_bracket = self._create_bracket(s)
                cur_band.append(cur_bracket)
                self._state['bracket'] = cur_bracket
        self._state['bracket'].add_trial(trial)
        self._trial_info[trial] = (cur_bracket, self._state['band_idx'])

    def _create_bracket(self, s):
        if False:
            return 10
        return _Bracket(time_attr=self._time_attr, max_trials=self._get_n0(s), init_t_attr=self._get_r0(s), max_t_attr=self._max_t_attr, eta=self._eta, s=s, stop_last_trials=self._stop_last_trials)

    def _cur_band_filled(self) -> bool:
        if False:
            print('Hello World!')
        'Checks if the current band is filled.\n\n        The size of the current band should be equal to s_max_1'
        cur_band = self._hyperbands[self._state['band_idx']]
        return len(cur_band) == self._s_max_1

    def on_trial_result(self, tune_controller: 'TuneController', trial: Trial, result: Dict):
        if False:
            return 10
        'If bracket is finished, all trials will be stopped.\n\n        If a given trial finishes and bracket iteration is not done,\n        the trial will be paused and resources will be given up.\n\n        This scheduler will not start trials but will stop trials.\n        The current running trial will not be handled,\n        as the trialrunner will be given control to handle it.'
        (bracket, _) = self._trial_info[trial]
        bracket.update_trial_stats(trial, result)
        if bracket.continue_trial(trial):
            return TrialScheduler.CONTINUE
        logger.debug(f'Processing bracket after trial {trial} result')
        action = self._process_bracket(tune_controller, bracket)
        logger.debug(f'{action} for {trial} on {self._time_attr}={result.get(self._time_attr)}')
        return action

    def _process_bracket(self, tune_controller: 'TuneController', bracket: '_Bracket') -> str:
        if False:
            while True:
                i = 10
        'This is called whenever a trial makes progress.\n\n        When all live trials in the bracket have no more iterations left,\n        Trials will be successively halved. If bracket is done, all\n        non-running trials will be stopped and cleaned up,\n        and during each halving phase, bad trials will be stopped while good\n        trials will return to "PENDING".\n\n        Note some implicit conditions here: In ``on_trial_result`` a trial is\n        either continued (e.g. if it didn\'t reach the time threshold for the bracket)\n        or this method (``_process_bracket``) is called. If there are other trials left\n        that still haven\'t reached the threshold, the trial is PAUSED. This means\n        that when the bracket is actually processed (``bracket.cur_iter_done``), there\n        is at most one RUNNING trial (which is the trial that is currently processed)\n        and the rest are either PAUSED (as explained above) or TERMINATED/ERRORED\n        (if they finish separately).\n        '
        action = TrialScheduler.PAUSE
        if bracket.cur_iter_done():
            if bracket.finished():
                bracket.cleanup_full(tune_controller)
                return TrialScheduler.STOP
            bracket.is_being_processed = True
            (good, bad) = bracket.successive_halving(self._metric, self._metric_op)
            logger.debug(f'Processing {len(good)} good and {len(bad)} bad trials in bracket {bracket}.\nGood: {good}\nBad: {bad}')
            self._num_stopped += len(bad)
            for t in bad:
                if t.status == Trial.PAUSED or t.is_saving:
                    logger.debug(f'Stopping other trial {str(t)}')
                    tune_controller.stop_trial(t)
                elif t.status == Trial.RUNNING:
                    logger.debug(f'Stopping current trial {str(t)}')
                    bracket.cleanup_trial(t)
                    action = TrialScheduler.STOP
                else:
                    raise TuneError(f'Trial with unexpected bad status encountered: {str(t)} is {t.status}')
            for t in good:
                if bracket.continue_trial(t):
                    assert t.status not in (Trial.ERROR, Trial.TERMINATED), f'Good trial {t.trial_id} is in an invalid state: {t.status}\nExpected trial to be either PAUSED, PENDING, or RUNNING.\nIf you encounter this, please file an issue on the Ray Github.'
                    if t.status == Trial.PAUSED or t.is_saving:
                        logger.debug(f'Unpausing trial {str(t)}')
                        self._unpause_trial(tune_controller, t)
                        bracket.trials_to_unpause.add(t)
                    elif t.status == Trial.RUNNING:
                        logger.debug(f'Continuing current trial {str(t)}')
                        action = TrialScheduler.CONTINUE
                elif bracket.finished() and bracket.stop_last_trials:
                    if t.status == Trial.PAUSED or t.is_saving:
                        logger.debug(f'Bracket finished. Stopping other trial {str(t)}')
                        tune_controller.stop_trial(t)
                    elif t.status == Trial.RUNNING:
                        logger.debug(f'Bracket finished. Stopping current trial {str(t)}')
                        bracket.cleanup_trial(t)
                        action = TrialScheduler.STOP
        return action

    def _unpause_trial(self, tune_controller: 'TuneController', trial: Trial):
        if False:
            for i in range(10):
                print('nop')
        'No-op by default.'
        return

    def on_trial_remove(self, tune_controller: 'TuneController', trial: Trial):
        if False:
            for i in range(10):
                print('nop')
        'Notification when trial terminates.\n\n        Trial info is removed from bracket. Triggers halving if bracket is\n        not finished.'
        (bracket, _) = self._trial_info[trial]
        bracket.cleanup_trial(trial)
        if not bracket.finished() and (not bracket.is_being_processed):
            logger.debug(f'Processing bracket after trial {trial} removed')
            self._process_bracket(tune_controller, bracket)

    def on_trial_complete(self, tune_controller: 'TuneController', trial: Trial, result: Dict):
        if False:
            return 10
        'Cleans up trial info from bracket if trial completed early.'
        self.on_trial_remove(tune_controller, trial)

    def on_trial_error(self, tune_controller: 'TuneController', trial: Trial):
        if False:
            return 10
        'Cleans up trial info from bracket if trial errored early.'
        self.on_trial_remove(tune_controller, trial)

    def choose_trial_to_run(self, tune_controller: 'TuneController') -> Optional[Trial]:
        if False:
            while True:
                i = 10
        'Fair scheduling within iteration by completion percentage.\n\n        List of trials not used since all trials are tracked as state\n        of scheduler. If iteration is occupied (ie, no trials to run),\n        then look into next iteration.\n        '
        for hyperband in self._hyperbands:
            scrubbed = [b for b in hyperband if b is not None]
            for bracket in sorted(scrubbed, key=lambda b: b.completion_percentage()):
                for trial in bracket.current_trials():
                    if trial.status == Trial.PAUSED and trial in bracket.trials_to_unpause or trial.status == Trial.PENDING:
                        return trial
        return None

    def debug_string(self) -> str:
        if False:
            print('Hello World!')
        'This provides a progress notification for the algorithm.\n\n        For each bracket, the algorithm will output a string as follows:\n\n            Bracket(Max Size (n)=5, Milestone (r)=33, completed=14.6%):\n            {PENDING: 2, RUNNING: 3, TERMINATED: 2}\n\n        "Max Size" indicates the max number of pending/running experiments\n        set according to the Hyperband algorithm.\n\n        "Milestone" indicates the iterations a trial will run for before\n        the next halving will occur.\n\n        "Completed" indicates an approximate progress metric. Some brackets,\n        like ones that are unfilled, will not reach 100%.\n        '
        out = 'Using HyperBand: '
        out += 'num_stopped={} total_brackets={}'.format(self._num_stopped, sum((len(band) for band in self._hyperbands)))
        for (i, band) in enumerate(self._hyperbands):
            out += '\nRound #{}:'.format(i)
            for bracket in band:
                if bracket:
                    out += '\n  {}'.format(bracket)
        return out

    def state(self) -> Dict[str, int]:
        if False:
            print('Hello World!')
        return {'num_brackets': sum((len(band) for band in self._hyperbands)), 'num_stopped': self._num_stopped}

class _Bracket:
    """Logical object for tracking Hyperband bracket progress. Keeps track
    of proper parameters as designated by HyperBand.

    Also keeps track of progress to ensure good scheduling.
    """

    def __init__(self, time_attr: str, max_trials: int, init_t_attr: int, max_t_attr: int, eta: float, s: int, stop_last_trials: bool=True):
        if False:
            return 10
        self._live_trials = {}
        self._all_trials = []
        self._time_attr = time_attr
        self._n = self._n0 = max_trials
        self._r = self._r0 = init_t_attr
        self._max_t_attr = max_t_attr
        self._cumul_r = self._r0
        self._eta = eta
        self._halves = s
        self._total_work = self._calculate_total_work(self._n0, self._r0, s)
        self._completed_progress = 0
        self.stop_last_trials = stop_last_trials
        self.is_being_processed = False
        self.trials_to_unpause = set()

    def add_trial(self, trial: Trial):
        if False:
            return 10
        'Add trial to bracket assuming bracket is not filled.\n\n        At a later iteration, a newly added trial will be given equal\n        opportunity to catch up.'
        assert not self.filled(), 'Cannot add trial to filled bracket!'
        self._live_trials[trial] = None
        self._all_trials.append(trial)

    def cur_iter_done(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Checks if all iterations have completed.\n\n        TODO(rliaw): also check that `t.iterations == self._r`'
        return all((self._get_result_time(result) >= self._cumul_r for result in self._live_trials.values()))

    def finished(self) -> bool:
        if False:
            return 10
        if not self.stop_last_trials:
            return False
        return self._halves == 0 and self.cur_iter_done()

    def current_trials(self) -> List[Trial]:
        if False:
            for i in range(10):
                print('nop')
        return list(self._live_trials)

    def continue_trial(self, trial: Trial) -> bool:
        if False:
            return 10
        result = self._live_trials[trial]
        if not self.stop_last_trials and self._halves == 0:
            return True
        elif self._get_result_time(result) < self._cumul_r:
            logger.debug(f"Continuing trial {trial} as it hasn't reached the time threshold {self._cumul_r}, yet.")
            return True
        return False

    def filled(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if bracket is filled.\n\n        Only let new trials be added at current level minimizing the need\n        to backtrack and bookkeep previous medians.'
        return len(self._live_trials) == self._n

    def successive_halving(self, metric: str, metric_op: float) -> Tuple[List[Trial], List[Trial]]:
        if False:
            for i in range(10):
                print('nop')
        if self._halves == 0 and (not self.stop_last_trials):
            return (self._live_trials, [])
        assert self._halves > 0
        self._halves -= 1
        self._n = int(np.ceil(self._n / self._eta))
        self._r *= self._eta
        self._r = int(min(self._r, self._max_t_attr))
        self._cumul_r = self._r
        sorted_trials = sorted(self._live_trials, key=lambda t: metric_op * self._live_trials[t][metric])
        (good, bad) = (sorted_trials[-self._n:], sorted_trials[:-self._n])
        return (good, bad)

    def update_trial_stats(self, trial: Trial, result: Dict):
        if False:
            return 10
        "Update result for trial. Called after trial has finished\n        an iteration - will decrement iteration count.\n\n        TODO(rliaw): The other alternative is to keep the trials\n        in and make sure they're not set as pending later."
        assert trial in self._live_trials
        assert self._get_result_time(result) >= 0
        observed_time = self._get_result_time(result)
        last_observed = self._get_result_time(self._live_trials[trial])
        delta = observed_time - last_observed
        if delta <= 0:
            logger.info('Restoring from a previous point in time. Previous={}; Now={}'.format(last_observed, observed_time))
        self._completed_progress += delta
        self._live_trials[trial] = result
        self.trials_to_unpause.discard(trial)

    def cleanup_trial(self, trial: Trial):
        if False:
            while True:
                i = 10
        'Clean up statistics tracking for terminated trials (either by force\n        or otherwise).\n\n        This may cause bad trials to continue for a long time, in the case\n        where all the good trials finish early and there are only bad trials\n        left in a bracket with a large max-iteration.'
        self._live_trials.pop(trial, None)

    def cleanup_full(self, tune_controller: 'TuneController'):
        if False:
            return 10
        'Cleans up bracket after bracket is completely finished.\n\n        Lets the last trial continue to run until termination condition\n        kicks in.'
        for trial in self.current_trials():
            if trial.status == Trial.PAUSED:
                tune_controller.stop_trial(trial)

    def completion_percentage(self) -> float:
        if False:
            i = 10
            return i + 15
        'Returns a progress metric.\n\n        This will not be always finish with 100 since dead trials\n        are dropped.'
        if self.finished():
            return 1.0
        return min(self._completed_progress / self._total_work, 1.0)

    def _get_result_time(self, result: Dict) -> float:
        if False:
            return 10
        if result is None:
            return 0
        return result[self._time_attr]

    def _calculate_total_work(self, n: int, r: float, s: int):
        if False:
            return 10
        work = 0
        cumulative_r = r
        for _ in range(s + 1):
            work += int(n) * int(r)
            n /= self._eta
            n = int(np.ceil(n))
            r *= self._eta
            r = int(min(r, self._max_t_attr - cumulative_r))
        return work

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        status = ', '.join(['Max Size (n)={}'.format(self._n), 'Milestone (r)={}'.format(self._cumul_r), 'completed={:.1%}'.format(self.completion_percentage())])
        counts = collections.Counter([t.status for t in self._all_trials])
        trial_statuses = ', '.join(sorted(('{}: {}'.format(k, v) for (k, v) in counts.items())))
        return 'Bracket({}): {{{}}} '.format(status, trial_statuses)