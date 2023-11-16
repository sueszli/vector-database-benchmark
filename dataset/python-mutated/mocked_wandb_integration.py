from collections import namedtuple
from dataclasses import dataclass
from typing import Dict
from unittest.mock import Mock
from wandb.util import json_dumps_safer
import ray
from ray.air.integrations.wandb import _WandbLoggingActor, WandbLoggerCallback

class Trial(namedtuple('MockTrial', ['config', 'trial_id', 'trial_name', 'experiment_dir_name', 'placement_group_factory', 'local_path'])):

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.trial_id)

    def __str__(self):
        if False:
            print('Hello World!')
        return self.trial_name

@dataclass
class LoggingActorState:
    args: list
    kwargs: dict
    exclude: list
    logs: list
    config: dict

class _FakeConfig:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.config = {}

    def update(self, config, *args, **kwargs):
        if False:
            return 10
        self.config.update(config)

class _MockWandbAPI:
    """Thread-safe.

    Note: Not implemented to mock re-init behavior properly. Proceed with caution."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.logs = []
        self.config = _FakeConfig()

    def init(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        mock = Mock()
        mock.args = args
        mock.kwargs = kwargs
        if 'config' in kwargs:
            self.config.update(kwargs['config'])
        return mock

    def log(self, data):
        if False:
            for i in range(10):
                print('nop')
        try:
            json_dumps_safer(data)
        except Exception:
            self.logs.append('serialization error')
        else:
            self.logs.append(data)

    def finish(self):
        if False:
            print('Hello World!')
        pass

    def get_logs(self):
        if False:
            i = 10
            return i + 15
        return self.logs

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return self.config.config

class _MockWandbLoggingActor(_WandbLoggingActor):
    _mock_wandb_api_cls = _MockWandbAPI

    def __init__(self, logdir, queue, exclude, to_config, *args, **kwargs):
        if False:
            return 10
        super(_MockWandbLoggingActor, self).__init__(logdir, queue, exclude, to_config, *args, **kwargs)
        self._wandb = self._mock_wandb_api_cls()

    def get_state(self):
        if False:
            for i in range(10):
                print('nop')
        return LoggingActorState(args=self.args, kwargs=self.kwargs, exclude=self._exclude, logs=self._wandb.get_logs(), config=self._wandb.get_config())

class WandbTestExperimentLogger(WandbLoggerCallback):
    """Wandb logger with mocked Wandb API gateway (one per trial)."""
    _logger_actor_cls = _MockWandbLoggingActor

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self._saved_actor_states: Dict['Trial', LoggingActorState] = {}

    def _cleanup_logging_actor(self, trial: 'Trial', **kwargs):
        if False:
            print('Hello World!')
        logging_actor_state: LoggingActorState = ray.get(self._trial_logging_actors[trial].get_state.remote())
        self._saved_actor_states[trial] = logging_actor_state
        super()._cleanup_logging_actor(trial, **kwargs)

    @property
    def trial_logging_actor_states(self) -> Dict['Trial', LoggingActorState]:
        if False:
            for i in range(10):
                print('nop')
        return self._saved_actor_states

def get_mock_wandb_logger(mock_api_cls=_MockWandbAPI, **kwargs):
    if False:
        return 10

    class MockWandbLoggingActor(_MockWandbLoggingActor):
        _mock_wandb_api_cls = mock_api_cls
    logger = WandbTestExperimentLogger(project='test_project', api_key='1234', **kwargs)
    logger._logger_actor_cls = MockWandbLoggingActor
    return logger