from typing import List
from reactivex import abc, typing
from reactivex.scheduler import VirtualTimeScheduler

class MockDisposable(abc.DisposableBase):

    def __init__(self, scheduler: VirtualTimeScheduler):
        if False:
            for i in range(10):
                print('nop')
        self.scheduler = scheduler
        self.disposes: List[typing.AbsoluteTime] = []
        self.disposes.append(self.scheduler.clock)

    def dispose(self) -> None:
        if False:
            print('Hello World!')
        self.disposes.append(self.scheduler.clock)