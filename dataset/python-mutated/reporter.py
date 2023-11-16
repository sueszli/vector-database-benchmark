from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

class BaseReporter:

    def report_download(self, link: Any, completed: int, total: int | None) -> None:
        if False:
            print('Hello World!')
        pass

    def report_build_start(self, filename: str) -> None:
        if False:
            print('Hello World!')
        pass

    def report_build_end(self, filename: str) -> None:
        if False:
            print('Hello World!')
        pass

    def report_unpack(self, filename: str, completed: int, total: int | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

@dataclass
class RichProgressReporter(BaseReporter):
    progress: Progress
    task_id: TaskID

    def report_download(self, link: Any, completed: int, total: int | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.progress.update(self.task_id, completed=completed, total=total, text='Downloading...')

    def report_unpack(self, filename: str, completed: int, total: int | None) -> None:
        if False:
            print('Hello World!')
        self.progress.update(self.task_id, completed=completed, total=total, text='Unpacking...')

    def report_build_start(self, filename: str) -> None:
        if False:
            while True:
                i = 10
        task = self.progress._tasks[self.task_id]
        task.total = None
        self.progress.update(self.task_id, text='Building...')

    def report_build_end(self, filename: str) -> None:
        if False:
            return 10
        self.progress.update(self.task_id, text='')