"""Base executor - this is the base class for all the implemented executors."""
from __future__ import annotations
import logging
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple
import pendulum
from airflow.cli.cli_config import DefaultHelpParser
from airflow.configuration import conf
from airflow.exceptions import RemovedInAirflow3Warning
from airflow.stats import Stats
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.state import TaskInstanceState
PARALLELISM: int = conf.getint('core', 'PARALLELISM')
if TYPE_CHECKING:
    import argparse
    from datetime import datetime
    from airflow.callbacks.base_callback_sink import BaseCallbackSink
    from airflow.callbacks.callback_requests import CallbackRequest
    from airflow.cli.cli_config import GroupCommand
    from airflow.models.taskinstance import TaskInstance
    from airflow.models.taskinstancekey import TaskInstanceKey
    CommandType = List[str]
    QueuedTaskInstanceType = Tuple[CommandType, int, Optional[str], TaskInstance]
    EventBufferValueType = Tuple[Optional[str], Any]
    TaskTuple = Tuple[TaskInstanceKey, CommandType, Optional[str], Optional[Any]]
log = logging.getLogger(__name__)

@dataclass
class RunningRetryAttemptType:
    """
    For keeping track of attempts to queue again when task still apparently running.

    We don't want to slow down the loop, so we don't block, but we allow it to be
    re-checked for at least MIN_SECONDS seconds.
    """
    MIN_SECONDS = 10
    total_tries: int = field(default=0, init=False)
    tries_after_min: int = field(default=0, init=False)
    first_attempt_time: datetime = field(default_factory=lambda : pendulum.now('UTC'), init=False)

    @property
    def elapsed(self):
        if False:
            i = 10
            return i + 15
        'Seconds since first attempt.'
        return (pendulum.now('UTC') - self.first_attempt_time).total_seconds()

    def can_try_again(self):
        if False:
            print('Hello World!')
        'Return False if there has been at least one try greater than MIN_SECONDS, otherwise return True.'
        if self.tries_after_min > 0:
            return False
        self.total_tries += 1
        elapsed = self.elapsed
        if elapsed > self.MIN_SECONDS:
            self.tries_after_min += 1
        log.debug('elapsed=%s tries=%s', elapsed, self.total_tries)
        return True

class BaseExecutor(LoggingMixin):
    """
    Base class to inherit for concrete executors such as Celery, Kubernetes, Local, Sequential, etc.

    :param parallelism: how many jobs should run at one time. Set to ``0`` for infinity.
    """
    supports_ad_hoc_ti_run: bool = False
    supports_pickling: bool = True
    supports_sentry: bool = False
    is_local: bool = False
    is_single_threaded: bool = False
    is_production: bool = True
    change_sensor_mode_to_reschedule: bool = False
    serve_logs: bool = False
    job_id: None | int | str = None
    callback_sink: BaseCallbackSink | None = None

    def __init__(self, parallelism: int=PARALLELISM):
        if False:
            print('Hello World!')
        super().__init__()
        self.parallelism: int = parallelism
        self.queued_tasks: dict[TaskInstanceKey, QueuedTaskInstanceType] = {}
        self.running: set[TaskInstanceKey] = set()
        self.event_buffer: dict[TaskInstanceKey, EventBufferValueType] = {}
        self.attempts: dict[TaskInstanceKey, RunningRetryAttemptType] = defaultdict(RunningRetryAttemptType)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}(parallelism={self.parallelism})'

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Executors may need to get things started.'

    def queue_command(self, task_instance: TaskInstance, command: CommandType, priority: int=1, queue: str | None=None):
        if False:
            return 10
        'Queues command to task.'
        if task_instance.key not in self.queued_tasks:
            self.log.info('Adding to queue: %s', command)
            self.queued_tasks[task_instance.key] = (command, priority, queue, task_instance)
        else:
            self.log.error('could not queue task %s', task_instance.key)

    def queue_task_instance(self, task_instance: TaskInstance, mark_success: bool=False, pickle_id: int | None=None, ignore_all_deps: bool=False, ignore_depends_on_past: bool=False, wait_for_past_depends_before_skipping: bool=False, ignore_task_deps: bool=False, ignore_ti_state: bool=False, pool: str | None=None, cfg_path: str | None=None) -> None:
        if False:
            return 10
        'Queues task instance.'
        pool = pool or task_instance.pool
        command_list_to_run = task_instance.command_as_list(local=True, mark_success=mark_success, ignore_all_deps=ignore_all_deps, ignore_depends_on_past=ignore_depends_on_past, wait_for_past_depends_before_skipping=wait_for_past_depends_before_skipping, ignore_task_deps=ignore_task_deps, ignore_ti_state=ignore_ti_state, pool=pool, pickle_id=pickle_id, cfg_path=cfg_path)
        self.log.debug('created command %s', command_list_to_run)
        self.queue_command(task_instance, command_list_to_run, priority=task_instance.task.priority_weight_total, queue=task_instance.task.queue)

    def has_task(self, task_instance: TaskInstance) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if a task is either queued or running in this executor.\n\n        :param task_instance: TaskInstance\n        :return: True if the task is known to this executor\n        '
        return task_instance.key in self.queued_tasks or task_instance.key in self.running

    def sync(self) -> None:
        if False:
            return 10
        '\n        Sync will get called periodically by the heartbeat method.\n\n        Executors should override this to perform gather statuses.\n        '

    def heartbeat(self) -> None:
        if False:
            print('Hello World!')
        'Heartbeat sent to trigger new jobs.'
        if not self.parallelism:
            open_slots = len(self.queued_tasks)
        else:
            open_slots = self.parallelism - len(self.running)
        num_running_tasks = len(self.running)
        num_queued_tasks = len(self.queued_tasks)
        self.log.debug('%s running task instances', num_running_tasks)
        self.log.debug('%s in queue', num_queued_tasks)
        self.log.debug('%s open slots', open_slots)
        Stats.gauge('executor.open_slots', value=open_slots, tags={'status': 'open', 'name': self.__class__.__name__})
        Stats.gauge('executor.queued_tasks', value=num_queued_tasks, tags={'status': 'queued', 'name': self.__class__.__name__})
        Stats.gauge('executor.running_tasks', value=num_running_tasks, tags={'status': 'running', 'name': self.__class__.__name__})
        self.trigger_tasks(open_slots)
        self.log.debug('Calling the %s sync method', self.__class__)
        self.sync()

    def order_queued_tasks_by_priority(self) -> list[tuple[TaskInstanceKey, QueuedTaskInstanceType]]:
        if False:
            while True:
                i = 10
        '\n        Orders the queued tasks by priority.\n\n        :return: List of tuples from the queued_tasks according to the priority.\n        '
        return sorted(self.queued_tasks.items(), key=lambda x: x[1][1], reverse=True)

    def trigger_tasks(self, open_slots: int) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Initiate async execution of the queued tasks, up to the number of available slots.\n\n        :param open_slots: Number of open slots\n        '
        sorted_queue = self.order_queued_tasks_by_priority()
        task_tuples = []
        for _ in range(min((open_slots, len(self.queued_tasks)))):
            (key, (command, _, queue, ti)) = sorted_queue.pop(0)
            if key in self.running:
                attempt = self.attempts[key]
                if attempt.can_try_again():
                    self.log.info('queued but still running; attempt=%s task=%s', attempt.total_tries, key)
                    continue
                self.log.error('could not queue task %s (still running after %d attempts)', key, attempt.total_tries)
                del self.attempts[key]
                del self.queued_tasks[key]
            else:
                if key in self.attempts:
                    del self.attempts[key]
                task_tuples.append((key, command, queue, ti.executor_config))
        if task_tuples:
            self._process_tasks(task_tuples)

    def _process_tasks(self, task_tuples: list[TaskTuple]) -> None:
        if False:
            for i in range(10):
                print('nop')
        for (key, command, queue, executor_config) in task_tuples:
            del self.queued_tasks[key]
            self.execute_async(key=key, command=command, queue=queue, executor_config=executor_config)
            self.running.add(key)

    def change_state(self, key: TaskInstanceKey, state: TaskInstanceState, info=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Change state of the task.\n\n        :param info: Executor information for the task instance\n        :param key: Unique key for the task instance\n        :param state: State to set for the task.\n        '
        self.log.debug('Changing state: %s', key)
        try:
            self.running.remove(key)
        except KeyError:
            self.log.debug('Could not find key: %s', key)
        self.event_buffer[key] = (state, info)

    def fail(self, key: TaskInstanceKey, info=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set fail state for the event.\n\n        :param info: Executor information for the task instance\n        :param key: Unique key for the task instance\n        '
        self.change_state(key, TaskInstanceState.FAILED, info)

    def success(self, key: TaskInstanceKey, info=None) -> None:
        if False:
            print('Hello World!')
        '\n        Set success state for the event.\n\n        :param info: Executor information for the task instance\n        :param key: Unique key for the task instance\n        '
        self.change_state(key, TaskInstanceState.SUCCESS, info)

    def get_event_buffer(self, dag_ids=None) -> dict[TaskInstanceKey, EventBufferValueType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return and flush the event buffer.\n\n        In case dag_ids is specified it will only return and flush events\n        for the given dag_ids. Otherwise, it returns and flushes all events.\n\n        :param dag_ids: the dag_ids to return events for; returns all if given ``None``.\n        :return: a dict of events\n        '
        cleared_events: dict[TaskInstanceKey, EventBufferValueType] = {}
        if dag_ids is None:
            cleared_events = self.event_buffer
            self.event_buffer = {}
        else:
            for ti_key in list(self.event_buffer.keys()):
                if ti_key.dag_id in dag_ids:
                    cleared_events[ti_key] = self.event_buffer.pop(ti_key)
        return cleared_events

    def execute_async(self, key: TaskInstanceKey, command: CommandType, queue: str | None=None, executor_config: Any | None=None) -> None:
        if False:
            print('Hello World!')
        '\n        Execute the command asynchronously.\n\n        :param key: Unique key for the task instance\n        :param command: Command to run\n        :param queue: name of the queue\n        :param executor_config: Configuration passed to the executor.\n        '
        raise NotImplementedError()

    def get_task_log(self, ti: TaskInstance, try_number: int) -> tuple[list[str], list[str]]:
        if False:
            while True:
                i = 10
        '\n        Return the task logs.\n\n        :param ti: A TaskInstance object\n        :param try_number: current try_number to read log from\n        :return: tuple of logs and messages\n        '
        return ([], [])

    def end(self) -> None:
        if False:
            return 10
        'Wait synchronously for the previously submitted job to complete.'
        raise NotImplementedError()

    def terminate(self):
        if False:
            for i in range(10):
                print('nop')
        'Get called when the daemon receives a SIGTERM.'
        raise NotImplementedError()

    def cleanup_stuck_queued_tasks(self, tis: list[TaskInstance]) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Handle remnants of tasks that were failed because they were stuck in queued.\n\n        Tasks can get stuck in queued. If such a task is detected, it will be marked\n        as `UP_FOR_RETRY` if the task instance has remaining retries or marked as `FAILED`\n        if it doesn't.\n\n        :param tis: List of Task Instances to clean up\n        :return: List of readable task instances for a warning message\n        "
        raise NotImplementedError()

    def try_adopt_task_instances(self, tis: Sequence[TaskInstance]) -> Sequence[TaskInstance]:
        if False:
            i = 10
            return i + 15
        '\n        Try to adopt running task instances that have been abandoned by a SchedulerJob dying.\n\n        Anything that is not adopted will be cleared by the scheduler (and then become eligible for\n        re-scheduling)\n\n        :return: any TaskInstances that were unable to be adopted\n        '
        return tis

    @property
    def slots_available(self):
        if False:
            i = 10
            return i + 15
        'Number of new tasks this executor instance can accept.'
        if self.parallelism:
            return self.parallelism - len(self.running) - len(self.queued_tasks)
        else:
            return sys.maxsize

    @staticmethod
    def validate_command(command: list[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Back-compat method to Check if the command to execute is airflow command.\n\n        :param command: command to check\n        '
        warnings.warn('\n            The `validate_command` method is deprecated. Please use ``validate_airflow_tasks_run_command``\n            ', RemovedInAirflow3Warning, stacklevel=2)
        BaseExecutor.validate_airflow_tasks_run_command(command)

    @staticmethod
    def validate_airflow_tasks_run_command(command: list[str]) -> tuple[str | None, str | None]:
        if False:
            print('Hello World!')
        '\n        Check if the command to execute is airflow command.\n\n        Returns tuple (dag_id,task_id) retrieved from the command (replaced with None values if missing)\n        '
        if command[0:3] != ['airflow', 'tasks', 'run']:
            raise ValueError('The command must start with ["airflow", "tasks", "run"].')
        if len(command) > 3 and '--help' not in command:
            dag_id: str | None = None
            task_id: str | None = None
            for arg in command[3:]:
                if not arg.startswith('--'):
                    if dag_id is None:
                        dag_id = arg
                    else:
                        task_id = arg
                        break
            return (dag_id, task_id)
        return (None, None)

    def debug_dump(self):
        if False:
            return 10
        'Get called in response to SIGUSR2 by the scheduler.'
        self.log.info('executor.queued (%d)\n\t%s', len(self.queued_tasks), '\n\t'.join(map(repr, self.queued_tasks.items())))
        self.log.info('executor.running (%d)\n\t%s', len(self.running), '\n\t'.join(map(repr, self.running)))
        self.log.info('executor.event_buffer (%d)\n\t%s', len(self.event_buffer), '\n\t'.join(map(repr, self.event_buffer.items())))

    def send_callback(self, request: CallbackRequest) -> None:
        if False:
            i = 10
            return i + 15
        'Send callback for execution.\n\n        Provides a default implementation which sends the callback to the `callback_sink` object.\n\n        :param request: Callback request to be executed.\n        '
        if not self.callback_sink:
            raise ValueError('Callback sink is not ready.')
        self.callback_sink.send(request)

    @staticmethod
    def get_cli_commands() -> list[GroupCommand]:
        if False:
            print('Hello World!')
        'Vends CLI commands to be included in Airflow CLI.\n\n        Override this method to expose commands via Airflow CLI to manage this executor. This can\n        be commands to setup/teardown the executor, inspect state, etc.\n        Make sure to choose unique names for those commands, to avoid collisions.\n        '
        return []

    @classmethod
    def _get_parser(cls) -> argparse.ArgumentParser:
        if False:
            print('Hello World!')
        'Generate documentation; used by Sphinx argparse.\n\n        :meta private:\n        '
        from airflow.cli.cli_parser import AirflowHelpFormatter, _add_command
        parser = DefaultHelpParser(prog='airflow', formatter_class=AirflowHelpFormatter)
        subparsers = parser.add_subparsers(dest='subcommand', metavar='GROUP_OR_COMMAND')
        for group_command in cls.get_cli_commands():
            _add_command(subparsers, group_command)
        return parser