import os
from typing import Any, Dict, Optional, Union
from unittest.mock import Mock
import lightning.pytorch as pl
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers.logger import Logger
from tests_pytorch.helpers.runif import RunIf

class AllRankLogger(Logger):
    """Logger to test all-rank logging (i.e. not just rank 0).

    Logs are saved to local variable `logs`.

    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.logs = {}
        self.exp = object()

    def experiment(self) -> Any:
        if False:
            while True:
                i = 10
        return self.exp

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int]=None):
        if False:
            while True:
                i = 10
        self.logs.update(metrics)

    def version(self) -> Union[int, str]:
        if False:
            for i in range(10):
                print('nop')
        return 1

    def name(self) -> str:
        if False:
            print('Hello World!')
        return 'AllRank'

    def log_hyperparams(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        pass

class TestModel(BoringModel):
    log_name = 'rank-{rank}'

    def on_train_start(self):
        if False:
            print('Hello World!')
        self.log(self.log_name.format(rank=self.local_rank), 0)

    def on_train_end(self):
        if False:
            print('Hello World!')
        assert self.log_name.format(rank=self.local_rank) in self.logger.logs, 'Expected rank to be logged'

@RunIf(skip_windows=True)
def test_all_rank_logging_ddp_cpu(tmpdir):
    if False:
        return 10
    'Check that all ranks can be logged from.'
    model = TestModel()
    all_rank_logger = AllRankLogger()
    trainer = Trainer(accelerator='cpu', devices=2, strategy='ddp_spawn', default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=1, max_epochs=1, enable_model_summary=False, logger=all_rank_logger, log_every_n_steps=1)
    trainer.fit(model)

@RunIf(min_cuda_gpus=2)
def test_all_rank_logging_ddp_spawn(tmpdir):
    if False:
        while True:
            i = 10
    'Check that all ranks can be logged from.'
    model = TestModel()
    all_rank_logger = AllRankLogger()
    trainer = Trainer(strategy='ddp_spawn', accelerator='gpu', devices=2, default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=1, max_epochs=1, logger=all_rank_logger, enable_model_summary=False)
    trainer.fit(model)

def test_first_logger_call_in_subprocess(tmpdir):
    if False:
        i = 10
        return i + 15
    'Test that the Trainer does not call the logger too early.\n\n    Only when the worker processes are initialized do we have access to the rank and know which one is the main process.\n\n    '

    class LoggerCallsObserver(Callback):

        def setup(self, trainer, pl_module, stage):
            if False:
                print('Hello World!')
            assert not trainer.logger.method_calls
            assert not os.listdir(trainer.logger.save_dir)

        def on_train_start(self, trainer, pl_module):
            if False:
                print('Hello World!')
            assert trainer.logger.method_call
            trainer.logger.log_graph.assert_called_once()
    logger = Mock()
    logger.version = '0'
    logger.name = 'name'
    logger.save_dir = tmpdir
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=1, max_epochs=1, logger=logger, callbacks=[LoggerCallsObserver()])
    trainer.fit(model)

def test_logger_after_fit_predict_test_calls(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Make sure logger outputs are finalized after fit, prediction, and test calls.'

    class BufferLogger(Logger):

        def __init__(self):
            if False:
                return 10
            super().__init__()
            self.buffer = {}
            self.logs = {}

        def log_metrics(self, metrics: Dict[str, float], step: Optional[int]=None) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.buffer.update(metrics)

        def finalize(self, status: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.logs.update(self.buffer)
            self.buffer = {}

        @property
        def experiment(self) -> Any:
            if False:
                for i in range(10):
                    print('nop')
            return None

        @property
        def version(self) -> Union[int, str]:
            if False:
                return 10
            return 1

        @property
        def name(self) -> str:
            if False:
                while True:
                    i = 10
            return 'BufferLogger'

        def log_hyperparams(self, *args, **kwargs) -> None:
            if False:
                while True:
                    i = 10
            return None

    class LoggerCallsObserver(Callback):

        def on_fit_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
            if False:
                while True:
                    i = 10
            trainer.logger.log_metrics({'fit': 1})

        def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
            if False:
                while True:
                    i = 10
            trainer.logger.log_metrics({'validate': 1})

        def on_predict_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
            if False:
                while True:
                    i = 10
            trainer.logger.log_metrics({'predict': 1})

        def on_test_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
            if False:
                while True:
                    i = 10
            trainer.logger.log_metrics({'test': 1})
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=1, max_epochs=1, logger=BufferLogger(), callbacks=[LoggerCallsObserver()])
    assert not trainer.logger.logs
    trainer.fit(model)
    assert trainer.logger.logs == {'fit': 1, 'validate': 1}
    trainer.test(model)
    assert trainer.logger.logs == {'fit': 1, 'validate': 1, 'test': 1}
    trainer.predict(model)
    assert trainer.logger.logs == {'fit': 1, 'validate': 1, 'test': 1, 'predict': 1}