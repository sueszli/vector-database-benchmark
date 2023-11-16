import os
import tempfile
from ray.tune import Callback
from ray.tune.execution.tune_controller import TuneController

class TrialResultObserver(Callback):
    """Helper class to control runner.step() count."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._counter = 0
        self._last_counter = 0

    def reset(self):
        if False:
            while True:
                i = 10
        self._last_counter = self._counter

    def just_received_a_result(self):
        if False:
            i = 10
            return i + 15
        if self._last_counter == self._counter:
            return False
        else:
            self._last_counter = self._counter
            return True

    def on_trial_result(self, **kwargs):
        if False:
            return 10
        self._counter += 1

def create_tune_experiment_checkpoint(trials: list, **runner_kwargs) -> str:
    if False:
        while True:
            i = 10
    experiment_dir = tempfile.mkdtemp()
    runner_kwargs.setdefault('experiment_path', experiment_dir)
    orig_env = os.environ.copy()
    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1'
    try:
        runner = TuneController(**runner_kwargs)
        for trial in trials:
            runner.add_trial(trial)
        runner.checkpoint(force=True)
    finally:
        os.environ.clear()
        os.environ.update(orig_env)
    return experiment_dir