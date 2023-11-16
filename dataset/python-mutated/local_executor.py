"""
LocalExecutor.

.. seealso::
    For more information on how the LocalExecutor works, take a look at the guide:
    :ref:`executor:LocalExecutor`
"""
from __future__ import annotations
import contextlib
import logging
import os
import subprocess
from abc import abstractmethod
from multiprocessing import Manager, Process
from queue import Empty
from typing import TYPE_CHECKING, Any, Optional, Tuple
from setproctitle import getproctitle, setproctitle
from airflow import settings
from airflow.exceptions import AirflowException
from airflow.executors.base_executor import PARALLELISM, BaseExecutor
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.state import TaskInstanceState
if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager
    from queue import Queue
    from airflow.executors.base_executor import CommandType
    from airflow.models.taskinstance import TaskInstanceStateType
    from airflow.models.taskinstancekey import TaskInstanceKey
    ExecutorWorkType = Tuple[Optional[TaskInstanceKey], Optional[CommandType]]

class LocalWorkerBase(Process, LoggingMixin):
    """
    LocalWorkerBase implementation to run airflow commands.

    Executes the given command and puts the result into a result queue when done, terminating execution.

    :param result_queue: the queue to store result state
    """

    def __init__(self, result_queue: Queue[TaskInstanceStateType]):
        if False:
            return 10
        super().__init__(target=self.do_work)
        self.daemon: bool = True
        self.result_queue: Queue[TaskInstanceStateType] = result_queue

    def run(self):
        if False:
            print('Hello World!')
        settings.engine.pool.dispose()
        settings.engine.dispose()
        setproctitle('airflow worker -- LocalExecutor')
        return super().run()

    def execute_work(self, key: TaskInstanceKey, command: CommandType) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Execute command received and stores result state in queue.\n\n        :param key: the key to identify the task instance\n        :param command: the command to execute\n        '
        if key is None:
            return
        self.log.info('%s running %s', self.__class__.__name__, command)
        setproctitle(f'airflow worker -- LocalExecutor: {command}')
        if settings.EXECUTE_TASKS_NEW_PYTHON_INTERPRETER:
            state = self._execute_work_in_subprocess(command)
        else:
            state = self._execute_work_in_fork(command)
        self.result_queue.put((key, state))
        setproctitle('airflow worker -- LocalExecutor')

    def _execute_work_in_subprocess(self, command: CommandType) -> TaskInstanceState:
        if False:
            i = 10
            return i + 15
        try:
            subprocess.check_call(command, close_fds=True)
            return TaskInstanceState.SUCCESS
        except subprocess.CalledProcessError as e:
            self.log.error('Failed to execute task %s.', e)
            return TaskInstanceState.FAILED

    def _execute_work_in_fork(self, command: CommandType) -> TaskInstanceState:
        if False:
            for i in range(10):
                print('nop')
        pid = os.fork()
        if pid:
            (pid, ret) = os.waitpid(pid, 0)
            return TaskInstanceState.SUCCESS if ret == 0 else TaskInstanceState.FAILED
        from airflow.sentry import Sentry
        ret = 1
        try:
            import signal
            from airflow.cli.cli_parser import get_parser
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGUSR2, signal.SIG_DFL)
            parser = get_parser()
            args = parser.parse_args(command[1:])
            args.shut_down_logging = False
            setproctitle(f'airflow task supervisor: {command}')
            args.func(args)
            ret = 0
            return TaskInstanceState.SUCCESS
        except Exception as e:
            self.log.exception('Failed to execute task %s.', e)
            return TaskInstanceState.FAILED
        finally:
            Sentry.flush()
            logging.shutdown()
            os._exit(ret)

    @abstractmethod
    def do_work(self):
        if False:
            while True:
                i = 10
        'Execute tasks; called in the subprocess.'
        raise NotImplementedError()

class LocalWorker(LocalWorkerBase):
    """
    Local worker that executes the task.

    :param result_queue: queue where results of the tasks are put.
    :param key: key identifying task instance
    :param command: Command to execute
    """

    def __init__(self, result_queue: Queue[TaskInstanceStateType], key: TaskInstanceKey, command: CommandType):
        if False:
            while True:
                i = 10
        super().__init__(result_queue)
        self.key: TaskInstanceKey = key
        self.command: CommandType = command

    def do_work(self) -> None:
        if False:
            i = 10
            return i + 15
        self.execute_work(key=self.key, command=self.command)

class QueuedLocalWorker(LocalWorkerBase):
    """
    LocalWorker implementation that is waiting for tasks from a queue.

    Will continue executing commands as they become available in the queue.
    It will terminate execution once the poison token is found.

    :param task_queue: queue from which worker reads tasks
    :param result_queue: queue where worker puts results after finishing tasks
    """

    def __init__(self, task_queue: Queue[ExecutorWorkType], result_queue: Queue[TaskInstanceStateType]):
        if False:
            print('Hello World!')
        super().__init__(result_queue=result_queue)
        self.task_queue = task_queue

    def do_work(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        while True:
            try:
                (key, command) = self.task_queue.get()
            except EOFError:
                self.log.info('Failed to read tasks from the task queue because the other end has closed the connection. Terminating worker %s.', self.name)
                break
            try:
                if key is None or command is None:
                    break
                self.execute_work(key=key, command=command)
            finally:
                self.task_queue.task_done()

class LocalExecutor(BaseExecutor):
    """
    LocalExecutor executes tasks locally in parallel.

    It uses the multiprocessing Python library and queues to parallelize the execution of tasks.

    :param parallelism: how many parallel processes are run in the executor
    """
    is_local: bool = True
    supports_pickling: bool = False
    serve_logs: bool = True

    def __init__(self, parallelism: int=PARALLELISM):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parallelism=parallelism)
        if self.parallelism < 0:
            raise AirflowException('parallelism must be bigger than or equal to 0')
        self.manager: SyncManager | None = None
        self.result_queue: Queue[TaskInstanceStateType] | None = None
        self.workers: list[QueuedLocalWorker] = []
        self.workers_used: int = 0
        self.workers_active: int = 0
        self.impl: None | (LocalExecutor.UnlimitedParallelism | LocalExecutor.LimitedParallelism) = None

    class UnlimitedParallelism:
        """
        Implement LocalExecutor with unlimited parallelism, starting one process per command executed.

        :param executor: the executor instance to implement.
        """

        def __init__(self, executor: LocalExecutor):
            if False:
                i = 10
                return i + 15
            self.executor: LocalExecutor = executor

        def start(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            'Start the executor.'
            self.executor.workers_used = 0
            self.executor.workers_active = 0

        def execute_async(self, key: TaskInstanceKey, command: CommandType, queue: str | None=None, executor_config: Any | None=None) -> None:
            if False:
                while True:
                    i = 10
            '\n            Execute task asynchronously.\n\n            :param key: the key to identify the task instance\n            :param command: the command to execute\n            :param queue: Name of the queue\n            :param executor_config: configuration for the executor\n            '
            if TYPE_CHECKING:
                assert self.executor.result_queue
            local_worker = LocalWorker(self.executor.result_queue, key=key, command=command)
            self.executor.workers_used += 1
            self.executor.workers_active += 1
            local_worker.start()

        def sync(self) -> None:
            if False:
                print('Hello World!')
            'Sync will get called periodically by the heartbeat method.'
            if not self.executor.result_queue:
                raise AirflowException('Executor should be started first')
            while not self.executor.result_queue.empty():
                results = self.executor.result_queue.get()
                self.executor.change_state(*results)
                self.executor.workers_active -= 1

        def end(self) -> None:
            if False:
                print('Hello World!')
            'Wait synchronously for the previously submitted job to complete.'
            while self.executor.workers_active > 0:
                self.executor.sync()

    class LimitedParallelism:
        """
        Implements LocalExecutor with limited parallelism.

        Uses a task queue to coordinate work distribution.

        :param executor: the executor instance to implement.
        """

        def __init__(self, executor: LocalExecutor):
            if False:
                print('Hello World!')
            self.executor: LocalExecutor = executor
            self.queue: Queue[ExecutorWorkType] | None = None

        def start(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            'Start limited parallelism implementation.'
            if TYPE_CHECKING:
                assert self.executor.manager
                assert self.executor.result_queue
            self.queue = self.executor.manager.Queue()
            self.executor.workers = [QueuedLocalWorker(self.queue, self.executor.result_queue) for _ in range(self.executor.parallelism)]
            self.executor.workers_used = len(self.executor.workers)
            for worker in self.executor.workers:
                worker.start()

        def execute_async(self, key: TaskInstanceKey, command: CommandType, queue: str | None=None, executor_config: Any | None=None) -> None:
            if False:
                i = 10
                return i + 15
            '\n            Execute task asynchronously.\n\n            :param key: the key to identify the task instance\n            :param command: the command to execute\n            :param queue: name of the queue\n            :param executor_config: configuration for the executor\n            '
            if TYPE_CHECKING:
                assert self.queue
            self.queue.put((key, command))

        def sync(self):
            if False:
                print('Hello World!')
            'Sync will get called periodically by the heartbeat method.'
            with contextlib.suppress(Empty):
                while True:
                    results = self.executor.result_queue.get_nowait()
                    try:
                        self.executor.change_state(*results)
                    finally:
                        self.executor.result_queue.task_done()

        def end(self):
            if False:
                while True:
                    i = 10
            '\n            End the executor.\n\n            Sends the poison pill to all workers.\n            '
            for _ in self.executor.workers:
                self.queue.put((None, None))
            self.queue.join()
            self.executor.sync()

    def start(self) -> None:
        if False:
            while True:
                i = 10
        'Start the executor.'
        old_proctitle = getproctitle()
        setproctitle('airflow executor -- LocalExecutor')
        self.manager = Manager()
        setproctitle(old_proctitle)
        self.result_queue = self.manager.Queue()
        self.workers = []
        self.workers_used = 0
        self.workers_active = 0
        self.impl = LocalExecutor.UnlimitedParallelism(self) if self.parallelism == 0 else LocalExecutor.LimitedParallelism(self)
        self.impl.start()

    def execute_async(self, key: TaskInstanceKey, command: CommandType, queue: str | None=None, executor_config: Any | None=None) -> None:
        if False:
            print('Hello World!')
        'Execute asynchronously.'
        if TYPE_CHECKING:
            assert self.impl
        self.validate_airflow_tasks_run_command(command)
        self.impl.execute_async(key=key, command=command, queue=queue, executor_config=executor_config)

    def sync(self) -> None:
        if False:
            return 10
        'Sync will get called periodically by the heartbeat method.'
        if TYPE_CHECKING:
            assert self.impl
        self.impl.sync()

    def end(self) -> None:
        if False:
            print('Hello World!')
        'End the executor.'
        if TYPE_CHECKING:
            assert self.impl
            assert self.manager
        self.log.info("Shutting down LocalExecutor; waiting for running tasks to finish.  Signal again if you don't want to wait.")
        self.impl.end()
        self.manager.shutdown()

    def terminate(self):
        if False:
            while True:
                i = 10
        'Terminate the executor is not doing anything.'