import logging
import os
import signal
import sys
import threading
from subprocess import call
from types import FrameType
from typing import Any, Callable, Dict, List, Set, Union
import lightning.pytorch as pl
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric.utilities.imports import _IS_WINDOWS, _PYTHON_GREATER_EQUAL_3_8_0
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message, rank_zero_info
_SIGNUM = Union[int, signal.Signals]
_HANDLER = Union[Callable[[_SIGNUM, FrameType], Any], int, signal.Handlers, None]
log = logging.getLogger(__name__)

class _HandlersCompose:

    def __init__(self, signal_handlers: Union[List[_HANDLER], _HANDLER]) -> None:
        if False:
            return 10
        if not isinstance(signal_handlers, list):
            signal_handlers = [signal_handlers]
        self.signal_handlers = signal_handlers

    def __call__(self, signum: _SIGNUM, frame: FrameType) -> None:
        if False:
            for i in range(10):
                print('nop')
        for signal_handler in self.signal_handlers:
            if isinstance(signal_handler, int):
                signal_handler = signal.getsignal(signal_handler)
            if callable(signal_handler):
                signal_handler(signum, frame)

class _SignalConnector:

    def __init__(self, trainer: 'pl.Trainer') -> None:
        if False:
            for i in range(10):
                print('nop')
        self.received_sigterm = False
        self.trainer = trainer
        self._original_handlers: Dict[_SIGNUM, _HANDLER] = {}

    def register_signal_handlers(self) -> None:
        if False:
            return 10
        self.received_sigterm = False
        self._original_handlers = self._get_current_signal_handlers()
        sigusr_handlers: List[_HANDLER] = []
        sigterm_handlers: List[_HANDLER] = [self._sigterm_notifier_fn]
        environment = self.trainer._accelerator_connector.cluster_environment
        if isinstance(environment, SLURMEnvironment) and environment.auto_requeue:
            log.info('SLURM auto-requeueing enabled. Setting signal handlers.')
            sigusr_handlers.append(self._slurm_sigusr_handler_fn)
            sigterm_handlers.append(self._sigterm_handler_fn)
        if not self._is_on_windows():
            sigusr = environment.requeue_signal if isinstance(environment, SLURMEnvironment) else signal.SIGUSR1
            assert sigusr is not None
            if sigusr_handlers and (not self._has_already_handler(sigusr)):
                self._register_signal(sigusr, _HandlersCompose(sigusr_handlers))
            if self._has_already_handler(signal.SIGTERM):
                sigterm_handlers.append(signal.getsignal(signal.SIGTERM))
            self._register_signal(signal.SIGTERM, _HandlersCompose(sigterm_handlers))

    def _slurm_sigusr_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        if False:
            for i in range(10):
                print('nop')
        rank_zero_info(f'Handling auto-requeue signal: {signum}')
        for logger in self.trainer.loggers:
            logger.finalize('finished')
        hpc_save_path = self.trainer._checkpoint_connector.hpc_save_path(self.trainer.default_root_dir)
        self.trainer.save_checkpoint(hpc_save_path)
        if self.trainer.is_global_zero:
            array_job_id = os.getenv('SLURM_ARRAY_JOB_ID')
            if array_job_id is not None:
                array_task_id = os.environ['SLURM_ARRAY_TASK_ID']
                job_id = f'{array_job_id}_{array_task_id}'
            else:
                job_id = os.environ['SLURM_JOB_ID']
            cmd = ['scontrol', 'requeue', job_id]
            log.info(f'requeing job {job_id}...')
            try:
                result = call(cmd)
            except FileNotFoundError:
                joint_cmd = [str(x) for x in cmd]
                result = call(' '.join(joint_cmd), shell=True)
            if result == 0:
                log.info(f'requeued exp {job_id}')
            else:
                log.warning('requeue failed...')

    def _sigterm_notifier_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        if False:
            i = 10
            return i + 15
        log.info(rank_prefixed_message(f'Received SIGTERM: {signum}', self.trainer.local_rank))
        if not self.received_sigterm:
            launcher = self.trainer.strategy.launcher
            if launcher is not None:
                launcher.kill(signum)
        self.received_sigterm = True

    def _sigterm_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        if False:
            for i in range(10):
                print('nop')
        log.info(f'Bypassing SIGTERM: {signum}')

    def teardown(self) -> None:
        if False:
            i = 10
            return i + 15
        'Restores the signals that were previously configured before :class:`_SignalConnector` replaced them.'
        for (signum, handler) in self._original_handlers.items():
            if handler is not None:
                self._register_signal(signum, handler)
        self._original_handlers = {}

    @staticmethod
    def _get_current_signal_handlers() -> Dict[_SIGNUM, _HANDLER]:
        if False:
            for i in range(10):
                print('nop')
        'Collects the currently assigned signal handlers.'
        valid_signals = _SignalConnector._valid_signals()
        if not _IS_WINDOWS:
            valid_signals -= {signal.SIGKILL, signal.SIGSTOP}
        return {signum: signal.getsignal(signum) for signum in valid_signals}

    @staticmethod
    def _valid_signals() -> Set[signal.Signals]:
        if False:
            while True:
                i = 10
        'Returns all valid signals supported on the current platform.\n\n        Behaves identically to :func:`signals.valid_signals` in Python 3.8+ and implements the equivalent behavior for\n        older Python versions.\n\n        '
        if _PYTHON_GREATER_EQUAL_3_8_0:
            return signal.valid_signals()
        if _IS_WINDOWS:
            return {signal.SIGABRT, signal.SIGFPE, signal.SIGILL, signal.SIGINT, signal.SIGSEGV, signal.SIGTERM, signal.SIGBREAK}
        return set(signal.Signals)

    @staticmethod
    def _is_on_windows() -> bool:
        if False:
            print('Hello World!')
        return sys.platform == 'win32'

    @staticmethod
    def _has_already_handler(signum: _SIGNUM) -> bool:
        if False:
            i = 10
            return i + 15
        return signal.getsignal(signum) not in (None, signal.SIG_DFL)

    @staticmethod
    def _register_signal(signum: _SIGNUM, handlers: _HANDLER) -> None:
        if False:
            return 10
        if threading.current_thread() is threading.main_thread():
            signal.signal(signum, handlers)

    def __getstate__(self) -> Dict:
        if False:
            print('Hello World!')
        state = self.__dict__.copy()
        state['_original_handlers'] = {}
        return state