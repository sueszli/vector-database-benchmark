from __future__ import annotations
from typing import Callable
from . import background_tasks

class AwaitableResponse:

    def __init__(self, fire_and_forget: Callable, wait_for_result: Callable) -> None:
        if False:
            while True:
                i = 10
        'Awaitable Response\n\n        This class can be used to run one of two different callables, depending on whether the response is awaited or not.\n        It must be awaited immediately after creation or not at all.\n\n        :param fire_and_forget: The callable to run if the response is not awaited.\n        :param wait_for_result: The callable to run if the response is awaited.\n        '
        self.fire_and_forget = fire_and_forget
        self.wait_for_result = wait_for_result
        self._is_fired = False
        self._is_awaited = False
        background_tasks.create(self._fire(), name='fire')

    async def _fire(self) -> None:
        if self._is_awaited:
            return
        self._is_fired = True
        self.fire_and_forget()

    def __await__(self):
        if False:
            i = 10
            return i + 15
        if self._is_fired:
            raise RuntimeError('AwaitableResponse must be awaited immediately after creation or not at all')
        self._is_awaited = True
        return self.wait_for_result().__await__()

class NullResponse(AwaitableResponse):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Null Response\n\n        This class can be used to create an AwaitableResponse that does nothing.\n        In contrast to AwaitableResponse, it can be created without a running event loop.\n        '

    def __await__(self):
        if False:
            while True:
                i = 10
        yield from []