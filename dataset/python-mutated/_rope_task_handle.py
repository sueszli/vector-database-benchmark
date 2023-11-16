from __future__ import annotations
import logging
from typing import Callable, ContextManager, List, Optional, Sequence
from rope.base.taskhandle import BaseJobSet, BaseTaskHandle
from pylsp.workspace import Workspace
log = logging.getLogger(__name__)
Report = Callable[[str, int], None]

class PylspJobSet(BaseJobSet):
    count: int = 0
    done: int = 0
    _reporter: Report
    _report_iter: ContextManager
    job_name: str = ''

    def __init__(self, count: Optional[int], report_iter: ContextManager):
        if False:
            while True:
                i = 10
        if count is not None:
            self.count = count
        self._reporter = report_iter.__enter__()
        self._report_iter = report_iter

    def started_job(self, name: Optional[str]) -> None:
        if False:
            print('Hello World!')
        if name:
            self.job_name = name

    def finished_job(self) -> None:
        if False:
            print('Hello World!')
        self.done += 1
        if self.get_percent_done() is not None and int(self.get_percent_done()) >= 100:
            if self._report_iter is None:
                return
            self._report_iter.__exit__(None, None, None)
            self._report_iter = None
        else:
            self._report()

    def check_status(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def get_percent_done(self) -> Optional[float]:
        if False:
            print('Hello World!')
        if self.count == 0:
            return 0
        return self.done / self.count * 100

    def increment(self) -> None:
        if False:
            return 10
        '\n        Increment the number of tasks to complete.\n\n        This is used if the number is not known ahead of time.\n        '
        self.count += 1
        self._report()

    def _report(self):
        if False:
            return 10
        percent = int(self.get_percent_done())
        message = f'{self.job_name} {self.done}/{self.count}'
        log.debug(f'Reporting {message} {percent}%')
        self._reporter(message, percent)

class PylspTaskHandle(BaseTaskHandle):
    name: str
    observers: List
    job_sets: List[PylspJobSet]
    stopped: bool
    workspace: Workspace
    _report: Callable[[str, str], None]

    def __init__(self, workspace: Workspace):
        if False:
            print('Hello World!')
        self.workspace = workspace
        self.job_sets = []
        self.observers = []

    def create_jobset(self, name='JobSet', count: Optional[int]=None):
        if False:
            print('Hello World!')
        report_iter = self.workspace.report_progress(name, None, None, skip_token_initialization=True)
        result = PylspJobSet(count, report_iter)
        self.job_sets.append(result)
        self._inform_observers()
        return result

    def stop(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def current_jobset(self) -> Optional[BaseJobSet]:
        if False:
            return 10
        pass

    def add_observer(self) -> None:
        if False:
            return 10
        pass

    def is_stopped(self) -> bool:
        if False:
            print('Hello World!')
        pass

    def get_jobsets(self) -> Sequence[BaseJobSet]:
        if False:
            while True:
                i = 10
        pass

    def _inform_observers(self) -> None:
        if False:
            i = 10
            return i + 15
        for observer in self.observers:
            observer()