"""OCRmyPDF concurrency abstractions."""
from __future__ import annotations
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Callable, TypeVar
from ocrmypdf._progressbar import NullProgressBar, ProgressBar
T = TypeVar('T')

def _task_noop(*_args, **_kwargs):
    if False:
        i = 10
        return i + 15
    return

class Executor(ABC):
    """Abstract concurrent executor."""
    pool_lock = threading.Lock()
    pbar_class = NullProgressBar

    def __init__(self, *, pbar_class=None):
        if False:
            while True:
                i = 10
        if pbar_class:
            self.pbar_class = pbar_class

    def __call__(self, *, use_threads: bool, max_workers: int, progress_kwargs: dict, worker_initializer: Callable | None=None, task: Callable[..., T] | None=None, task_arguments: Iterable | None=None, task_finished: Callable[[T, ProgressBar], None] | None=None) -> None:
        if False:
            print('Hello World!')
        "Set up parallel execution and progress reporting.\n\n        Args:\n            use_threads: If ``False``, the workload is the sort that will benefit from\n                running in a multiprocessing context (for example, it uses Python\n                heavily, and parallelizing it with threads is not expected to be\n                performant).\n            max_workers: The maximum number of workers that should be run.\n            progress_kwargs: Arguments to set up the progress bar.\n            worker_initializer: Called when a worker is initialized, in the worker's\n                execution context. If the child workers are processes, it must be\n                possible to marshall/pickle the worker initializer.\n                ``functools.partial`` can be used to bind parameters.\n            task: Called when the worker starts a new task, in the worker's execution\n                context. Must be possible to marshall to the worker.\n            task_finished: Called when a worker finishes a task, in the parent's\n                context.\n            task_arguments: An iterable that generates a group of parameters for each\n                task. This runs in the parent's context, but the parameters must be\n                marshallable to the worker.\n        "
        if not task_arguments:
            return
        if not worker_initializer:
            worker_initializer = _task_noop
        if not task_finished:
            task_finished = _task_noop
        if not task:
            task = _task_noop
        with self.pool_lock:
            self._execute(use_threads=use_threads, max_workers=max_workers, progress_kwargs=progress_kwargs, worker_initializer=worker_initializer, task=task, task_arguments=task_arguments, task_finished=task_finished)

    @abstractmethod
    def _execute(self, *, use_threads: bool, max_workers: int, progress_kwargs: dict, worker_initializer: Callable, task: Callable, task_arguments: Iterable, task_finished: Callable):
        if False:
            while True:
                i = 10
        'Custom executors should override this method.'

def setup_executor(plugin_manager) -> Executor:
    if False:
        print('Hello World!')
    pbar_class = plugin_manager.hook.get_progressbar_class()
    return plugin_manager.hook.get_executor(progressbar_class=pbar_class)

class SerialExecutor(Executor):
    """Implements a purely sequential executor using the parallel protocol.

    The current process/thread will be the worker that executes all tasks
    in order. As such, ``worker_initializer`` will never be called.
    """

    def _execute(self, *, use_threads: bool, max_workers: int, progress_kwargs: dict, worker_initializer: Callable, task: Callable, task_arguments: Iterable, task_finished: Callable):
        if False:
            print('Hello World!')
        with self.pbar_class(**progress_kwargs) as pbar:
            for args in task_arguments:
                result = task(*args)
                task_finished(result, pbar)