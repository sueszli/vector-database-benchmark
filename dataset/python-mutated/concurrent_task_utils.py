"""Utility methods for managing concurrent tasks."""
from __future__ import annotations
import datetime
import threading
import time
import traceback
from typing import Any, Callable, Final, List, Optional
LOG_LOCK: Final = threading.Lock()
ALL_ERRORS: Final = []
SUCCESS_MESSAGE_PREFIX: Final = 'SUCCESS '
FAILED_MESSAGE_PREFIX: Final = 'FAILED '

def log(message: str, show_time: bool=False) -> None:
    if False:
        return 10
    'Logs a message to the terminal.\n\n    If show_time is True, prefixes the message with the current time.\n    '
    with LOG_LOCK:
        if show_time:
            print(datetime.datetime.utcnow().strftime('%H:%M:%S'), message)
        else:
            print(message)

class TaskResult:
    """Task result for concurrent_task_utils."""

    def __init__(self, name: str, failed: bool, trimmed_messages: List[str], messages: List[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructs a TaskResult object.\n\n        Args:\n            name: str. The name of the task.\n            failed: bool. The boolean value representing whether the task\n                failed.\n            trimmed_messages: list(str). List of error messages that are\n                trimmed to keep main part of messages.\n            messages: list(str). List of full messages returned by the objects.\n        '
        self.name = name
        self.failed = failed
        self.trimmed_messages = trimmed_messages
        self.messages = messages

    def get_report(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of message with pass or fail status for the current\n        check.\n\n        Returns:\n            list(str). List of full messages corresponding to the given\n            task.\n        '
        all_messages = self.messages[:]
        status_message = '%s %s check %s' % ((FAILED_MESSAGE_PREFIX, self.name, 'failed') if self.failed else (SUCCESS_MESSAGE_PREFIX, self.name, 'passed'))
        all_messages.append(status_message)
        return all_messages

class TaskThread(threading.Thread):
    """Runs a task in its own thread."""

    def __init__(self, func: Callable[..., Any], verbose: bool, semaphore: threading.Semaphore, name: Optional[str], report_enabled: bool) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.func = func
        self.task_results: List[TaskResult] = []
        self.exception: Optional[Exception] = None
        self.stacktrace: Optional[str] = None
        self.verbose = verbose
        self.name = name
        self.semaphore = semaphore
        self.finished = False
        self.report_enabled = report_enabled

    def run(self) -> None:
        if False:
            print('Hello World!')
        try:
            self.task_results = self.func()
            if self.verbose:
                for task_result in self.task_results:
                    if self.report_enabled:
                        log('Report from %s check\n----------------------------------------\n%s' % (task_result.name, '\n'.join(task_result.get_report())), show_time=True)
                    else:
                        log('LOG %s:\n%s----------------------------------------' % (self.name, task_result.messages[0]), show_time=True)
            log('FINISHED %s: %.1f secs' % (self.name, time.time() - self.start_time), show_time=True)
        except Exception as e:
            self.exception = e
            self.stacktrace = traceback.format_exc()
            if 'KeyboardInterrupt' not in self.exception.args[0]:
                log(str(e))
                log('ERROR %s: %.1f secs' % (self.name, time.time() - self.start_time), show_time=True)
        finally:
            self.semaphore.release()
            self.finished = True

def _check_all_tasks(tasks: List[TaskThread]) -> None:
    if False:
        print('Hello World!')
    'Checks the results of all tasks.'
    running_tasks_data = []
    for task in tasks:
        if task.isAlive():
            running_tasks_data.append('  %s (started %s)' % (task.name, time.strftime('%H:%M:%S', time.localtime(task.start_time))))
        if task.exception:
            stacktrace = task.stacktrace if task.stacktrace else 'No stacktrace present.'
            ALL_ERRORS.append(stacktrace)
    if running_tasks_data:
        log('----------------------------------------')
        log('Tasks still running:')
        for task_details in running_tasks_data:
            log(task_details)

def execute_tasks(tasks: List[TaskThread], semaphore: threading.Semaphore) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Starts all tasks and checks the results.\n    Runs no more than the allowable limit defined in the semaphore.\n\n    Args:\n        tasks: list(TaskThread). The tasks to run.\n        semaphore: threading.Semaphore. The object that controls how many tasks\n            can run at any time.\n    '
    empty_tasks_list: List[TaskThread] = []
    remaining_tasks: List[TaskThread] = empty_tasks_list + tasks
    currently_running_tasks = []
    while remaining_tasks:
        task = remaining_tasks.pop()
        semaphore.acquire()
        task.start_time = time.time()
        task.start()
        currently_running_tasks.append(task)
        if len(remaining_tasks) % 5 == 0:
            if remaining_tasks:
                log('----------------------------------------')
                log('Number of unstarted tasks: %s' % len(remaining_tasks))
            _check_all_tasks(currently_running_tasks)
        log('----------------------------------------')
    for task in currently_running_tasks:
        task.join()
    _check_all_tasks(currently_running_tasks)

def create_task(func: Callable[..., Any], verbose: bool, semaphore: threading.Semaphore, name: Optional[str]=None, report_enabled: bool=True) -> TaskThread:
    if False:
        for i in range(10):
            print('nop')
    'Create a Task in its Thread.\n\n    Args:\n        func: Function. The function that is going to run.\n        verbose: bool. True if verbose mode is enabled.\n        semaphore: threading.Semaphore. The object that controls how many tasks\n            can run at any time.\n        name: str|None. Name of the task that is going to be created.\n        report_enabled: bool. Decide whether task result will print or not.\n\n    Returns:\n        task: TaskThread object. Created task.\n    '
    task = TaskThread(func, verbose, semaphore, name, report_enabled)
    return task