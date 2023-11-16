import inspect
import logging
import os
from functools import partial
from numbers import Number
from typing import Any, Callable, Dict, Optional, Type
from ray.air._internal.util import StartTraceback, RunnerThread
import queue
from ray.air.constants import _ERROR_FETCH_TIMEOUT
import ray.train
from ray.train._internal.checkpoint_manager import _TrainingResult
from ray.train._internal.session import init_session, get_session, shutdown_session, _TrainSession, TrialInfo
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.result import DEFAULT_METRIC, RESULT_DUPLICATE, SHOULD_CHECKPOINT
from ray.tune.trainable import Trainable
from ray.tune.utils import _detect_checkpoint_function, _detect_config_single, _detect_reporter
from ray.util.annotations import DeveloperAPI
logger = logging.getLogger(__name__)
NULL_MARKER = '.null_marker'
TEMP_MARKER = '.temp_marker'
_CHECKPOINT_DIR_ARG_DEPRECATION_MSG = 'Accepting a `checkpoint_dir` argument in your training function is deprecated.\nPlease use `ray.train.get_checkpoint()` to access your checkpoint as a\n`ray.train.Checkpoint` object instead. See below for an example:\n\nBefore\n------\n\nfrom ray import tune\n\ndef train_fn(config, checkpoint_dir=None):\n    if checkpoint_dir:\n        torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))\n    ...\n\ntuner = tune.Tuner(train_fn)\ntuner.fit()\n\nAfter\n-----\n\nfrom ray import train, tune\n\ndef train_fn(config):\n    checkpoint: train.Checkpoint = train.get_checkpoint()\n    if checkpoint:\n        with checkpoint.as_directory() as checkpoint_dir:\n            torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))\n    ...\n\ntuner = tune.Tuner(train_fn)\ntuner.fit()'
_REPORTER_ARG_DEPRECATION_MSG = 'Accepting a `reporter` in your training function is deprecated.\nPlease use `ray.train.report()` to report results instead. See below for an example:\n\nBefore\n------\n\nfrom ray import tune\n\ndef train_fn(config, reporter):\n    reporter(metric=1)\n\ntuner = tune.Tuner(train_fn)\ntuner.fit()\n\nAfter\n-----\n\nfrom ray import train, tune\n\ndef train_fn(config):\n    train.report({"metric": 1})\n\ntuner = tune.Tuner(train_fn)\ntuner.fit()'

@DeveloperAPI
class FunctionTrainable(Trainable):
    """Trainable that runs a user function reporting results.

    This mode of execution does not support checkpoint/restore."""
    _name = 'func'

    def setup(self, config):
        if False:
            i = 10
            return i + 15
        init_session(training_func=lambda : self._trainable_func(self.config), trial_info=TrialInfo(name=self.trial_name, id=self.trial_id, resources=self.trial_resources, logdir=self._storage.trial_local_path, driver_ip=None, experiment_name=self._storage.experiment_dir_name), storage=self._storage, synchronous_result_reporting=True, world_rank=None, local_rank=None, node_rank=None, local_world_size=None, world_size=None, dataset_shard=None, checkpoint=None)
        self._last_training_result: Optional[_TrainingResult] = None

    def _trainable_func(self, config: Dict[str, Any]):
        if False:
            while True:
                i = 10
        'Subclasses can override this to set the trainable func.'
        raise NotImplementedError

    def _start(self):
        if False:
            i = 10
            return i + 15

        def entrypoint():
            if False:
                for i in range(10):
                    print('nop')
            try:
                return self._trainable_func(self.config)
            except Exception as e:
                raise StartTraceback from e
        self._runner = RunnerThread(target=entrypoint, error_queue=self._error_queue, daemon=True)
        self._status_reporter._start()
        try:
            self._runner.start()
        except RuntimeError:
            pass

    def step(self):
        if False:
            return 10
        'Implements train() for a Function API.\n\n        If the RunnerThread finishes without reporting "done",\n        Tune will automatically provide a magic keyword __duplicate__\n        along with a result with "done=True". The TrialRunner will handle the\n        result accordingly (see tune/tune_controller.py).\n        '
        session: _TrainSession = get_session()
        if not session.training_started:
            session.start()
        training_result: Optional[_TrainingResult] = session.get_next()
        if not training_result:
            raise RuntimeError('Should not have reached here. The TuneController should not have scheduled another `train` remote call.It should have scheduled a `stop` instead after the training function exits.')
        metrics = training_result.metrics
        if RESULT_DUPLICATE in metrics:
            metrics[SHOULD_CHECKPOINT] = False
        self._last_training_result = training_result
        if training_result.checkpoint is not None:
            metrics[SHOULD_CHECKPOINT] = True
        return metrics

    def execute(self, fn):
        if False:
            i = 10
            return i + 15
        return fn(self)

    def save_checkpoint(self, checkpoint_dir: str=''):
        if False:
            for i in range(10):
                print('nop')
        if checkpoint_dir:
            raise ValueError('Checkpoint dir should not be used with function API.')
        return self._last_training_result

    def _create_checkpoint_dir(self, checkpoint_dir: Optional[str]=None) -> Optional[str]:
        if False:
            print('Hello World!')
        return None

    def load_checkpoint(self, checkpoint_result: _TrainingResult):
        if False:
            print('Hello World!')
        session = get_session()
        session.loaded_checkpoint = checkpoint_result.checkpoint

    def cleanup(self):
        if False:
            print('Hello World!')
        session = get_session()
        try:
            session.finish(timeout=0)
        finally:
            session._report_thread_runner_error()
            shutdown_session()

    def reset_config(self, new_config):
        if False:
            return 10
        session = get_session()
        thread_timeout = int(os.environ.get('TUNE_FUNCTION_THREAD_TIMEOUT_S', 2))
        session.finish(timeout=thread_timeout)
        if session.training_thread.is_alive():
            return False
        session.reset(training_func=lambda : self._trainable_func(self.config), trial_info=TrialInfo(name=self.trial_name, id=self.trial_id, resources=self.trial_resources, logdir=self._storage.trial_local_path, driver_ip=None, experiment_name=self._storage.experiment_dir_name), storage=self._storage)
        self._last_result = {}
        return True

    def _report_thread_runner_error(self, block=False):
        if False:
            while True:
                i = 10
        try:
            e = self._error_queue.get(block=block, timeout=_ERROR_FETCH_TIMEOUT)
            raise StartTraceback from e
        except queue.Empty:
            pass

@DeveloperAPI
def wrap_function(train_func: Callable[[Any], Any], warn: bool=True, name: Optional[str]=None) -> Type['FunctionTrainable']:
    if False:
        while True:
            i = 10
    inherit_from = (FunctionTrainable,)
    if hasattr(train_func, '__mixins__'):
        inherit_from = train_func.__mixins__ + inherit_from
    func_args = inspect.getfullargspec(train_func).args
    use_checkpoint = _detect_checkpoint_function(train_func)
    use_config_single = _detect_config_single(train_func)
    use_reporter = _detect_reporter(train_func)
    if use_checkpoint:
        raise DeprecationWarning(_CHECKPOINT_DIR_ARG_DEPRECATION_MSG)
    if use_reporter:
        raise DeprecationWarning(_REPORTER_ARG_DEPRECATION_MSG)
    if not use_config_single:
        raise ValueError("Unknown argument found in the Trainable function. The function args must include a 'config' positional parameter.Found: {}".format(func_args))
    resources = getattr(train_func, '_resources', None)

    class ImplicitFunc(*inherit_from):
        _name = name or (train_func.__name__ if hasattr(train_func, '__name__') else 'func')

        def __repr__(self):
            if False:
                print('Hello World!')
            return self._name

        def _trainable_func(self, config):
            if False:
                return 10
            fn = partial(train_func, config)

            def handle_output(output):
                if False:
                    i = 10
                    return i + 15
                if not output:
                    return
                elif isinstance(output, dict):
                    ray.train.report(output)
                elif isinstance(output, Number):
                    ray.train.report({DEFAULT_METRIC: output})
                else:
                    raise ValueError('Invalid return or yield value. Either return/yield a single number or a dictionary object in your trainable function.')
            output = None
            if inspect.isgeneratorfunction(train_func):
                for output in fn():
                    handle_output(output)
            else:
                output = fn()
                handle_output(output)
            ray.train.report({RESULT_DUPLICATE: True})
            return output

        @classmethod
        def default_resource_request(cls, config: Dict[str, Any]) -> Optional[PlacementGroupFactory]:
            if False:
                i = 10
                return i + 15
            if not isinstance(resources, PlacementGroupFactory) and callable(resources):
                return resources(config)
            return resources
    return ImplicitFunc