from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional
from reactivex import abc, typing
from .newthreadscheduler import NewThreadScheduler

class ThreadPoolScheduler(NewThreadScheduler):
    """A scheduler that schedules work via the thread pool."""

    class ThreadPoolThread(abc.StartableBase):
        """Wraps a concurrent future as a thread."""

        def __init__(self, executor: ThreadPoolExecutor, target: typing.StartableTarget):
            if False:
                print('Hello World!')
            self.executor: ThreadPoolExecutor = executor
            self.target: typing.StartableTarget = target
            self.future: Optional['Future[Any]'] = None

        def start(self) -> None:
            if False:
                print('Hello World!')
            self.future = self.executor.submit(self.target)

        def cancel(self) -> None:
            if False:
                print('Hello World!')
            if self.future:
                self.future.cancel()

    def __init__(self, max_workers: Optional[int]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)

        def thread_factory(target: typing.StartableTarget) -> ThreadPoolScheduler.ThreadPoolThread:
            if False:
                print('Hello World!')
            return self.ThreadPoolThread(self.executor, target)
        super().__init__(thread_factory)