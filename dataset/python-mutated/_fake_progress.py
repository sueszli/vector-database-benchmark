from typing import Any
from rich.progress import TaskID

class FakeProgress:
    """A fake progress bar that does nothing.

    This is used when the user has only one file to process."""

    def advance(self, task_id: TaskID) -> None:
        if False:
            while True:
                i = 10
        pass

    def add_task(self, *args: Any, **kwargs: Any) -> TaskID:
        if False:
            for i in range(10):
                print('nop')
        return TaskID(0)

    def __enter__(self) -> 'FakeProgress':
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *args: object, **kwargs: Any) -> None:
        if False:
            return 10
        pass