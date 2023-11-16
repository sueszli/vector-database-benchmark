import logging
from pathlib import Path
from typing import Any
from xgboost.callback import TrainingCallback
logger = logging.getLogger(__name__)

class BaseTensorboardLogger:

    def __init__(self, logdir: Path, activate: bool=True):
        if False:
            i = 10
            return i + 15
        pass

    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        if False:
            print('Hello World!')
        return

    def close(self):
        if False:
            print('Hello World!')
        return

class BaseTensorBoardCallback(TrainingCallback):

    def __init__(self, logdir: Path, activate: bool=True):
        if False:
            return 10
        pass

    def after_iteration(self, model, epoch: int, evals_log: TrainingCallback.EvalsLog) -> bool:
        if False:
            print('Hello World!')
        return False

    def after_training(self, model):
        if False:
            print('Hello World!')
        return model