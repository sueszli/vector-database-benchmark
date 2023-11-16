from typing import Dict, Optional, TYPE_CHECKING
from ray.air._internal.usage import tag_scheduler
from ray.tune.result import DEFAULT_METRIC
from ray.tune.experiment import Trial
from ray.util.annotations import DeveloperAPI, PublicAPI
if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController

@DeveloperAPI
class TrialScheduler:
    """Interface for implementing a Trial Scheduler class.

    Note to Tune developers: If a new scheduler is added, please update
    `air/_internal/usage.py`.
    """
    CONTINUE = 'CONTINUE'
    PAUSE = 'PAUSE'
    STOP = 'STOP'
    NOOP = 'NOOP'
    _metric = None
    _supports_buffered_results = True

    def __init__(self):
        if False:
            i = 10
            return i + 15
        tag_scheduler(self)

    @property
    def metric(self):
        if False:
            i = 10
            return i + 15
        return self._metric

    @property
    def supports_buffered_results(self):
        if False:
            return 10
        return self._supports_buffered_results

    def set_search_properties(self, metric: Optional[str], mode: Optional[str], **spec) -> bool:
        if False:
            print('Hello World!')
        'Pass search properties to scheduler.\n\n        This method acts as an alternative to instantiating schedulers\n        that react to metrics with their own `metric` and `mode` parameters.\n\n        Args:\n            metric: Metric to optimize\n            mode: One of ["min", "max"]. Direction to optimize.\n            **spec: Any kwargs for forward compatiblity.\n                Info like Experiment.PUBLIC_KEYS is provided through here.\n        '
        if self._metric and metric:
            return False
        if metric:
            self._metric = metric
        if self._metric is None:
            self._metric = DEFAULT_METRIC
        return True

    def on_trial_add(self, tune_controller: 'TuneController', trial: Trial):
        if False:
            return 10
        'Called when a new trial is added to the trial runner.'
        raise NotImplementedError

    def on_trial_error(self, tune_controller: 'TuneController', trial: Trial):
        if False:
            for i in range(10):
                print('nop')
        'Notification for the error of trial.\n\n        This will only be called when the trial is in the RUNNING state.'
        raise NotImplementedError

    def on_trial_result(self, tune_controller: 'TuneController', trial: Trial, result: Dict) -> str:
        if False:
            return 10
        'Called on each intermediate result returned by a trial.\n\n        At this point, the trial scheduler can make a decision by returning\n        one of CONTINUE, PAUSE, and STOP. This will only be called when the\n        trial is in the RUNNING state.'
        raise NotImplementedError

    def on_trial_complete(self, tune_controller: 'TuneController', trial: Trial, result: Dict):
        if False:
            i = 10
            return i + 15
        'Notification for the completion of trial.\n\n        This will only be called when the trial is in the RUNNING state and\n        either completes naturally or by manual termination.'
        raise NotImplementedError

    def on_trial_remove(self, tune_controller: 'TuneController', trial: Trial):
        if False:
            print('Hello World!')
        'Called to remove trial.\n\n        This is called when the trial is in PAUSED or PENDING state. Otherwise,\n        call `on_trial_complete`.'
        raise NotImplementedError

    def choose_trial_to_run(self, tune_controller: 'TuneController') -> Optional[Trial]:
        if False:
            i = 10
            return i + 15
        'Called to choose a new trial to run.\n\n        This should return one of the trials in tune_controller that is in\n        the PENDING or PAUSED state. This function must be idempotent.\n\n        If no trial is ready, return None.'
        raise NotImplementedError

    def debug_string(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns a human readable message for printing to the console.'
        raise NotImplementedError

    def save(self, checkpoint_path: str):
        if False:
            return 10
        'Save trial scheduler to a checkpoint'
        raise NotImplementedError

    def restore(self, checkpoint_path: str):
        if False:
            return 10
        'Restore trial scheduler from checkpoint.'
        raise NotImplementedError

@PublicAPI
class FIFOScheduler(TrialScheduler):
    """Simple scheduler that just runs trials in submission order."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def on_trial_add(self, tune_controller: 'TuneController', trial: Trial):
        if False:
            for i in range(10):
                print('nop')
        pass

    def on_trial_error(self, tune_controller: 'TuneController', trial: Trial):
        if False:
            print('Hello World!')
        pass

    def on_trial_result(self, tune_controller: 'TuneController', trial: Trial, result: Dict) -> str:
        if False:
            print('Hello World!')
        return TrialScheduler.CONTINUE

    def on_trial_complete(self, tune_controller: 'TuneController', trial: Trial, result: Dict):
        if False:
            return 10
        pass

    def on_trial_remove(self, tune_controller: 'TuneController', trial: Trial):
        if False:
            for i in range(10):
                print('nop')
        pass

    def choose_trial_to_run(self, tune_controller: 'TuneController') -> Optional[Trial]:
        if False:
            i = 10
            return i + 15
        for trial in tune_controller.get_trials():
            if trial.status == Trial.PENDING:
                return trial
        for trial in tune_controller.get_trials():
            if trial.status == Trial.PAUSED:
                return trial
        return None

    def debug_string(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'Using FIFO scheduling algorithm.'