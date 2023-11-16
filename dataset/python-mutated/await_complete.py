from __future__ import annotations
from asyncio import Future, gather
from typing import Any, Coroutine, Iterator, TypeVar
import rich.repr
ReturnType = TypeVar('ReturnType')

@rich.repr.auto(angular=True)
class AwaitComplete:
    """An 'optionally-awaitable' object."""

    def __init__(self, *coroutines: Coroutine[Any, Any, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Create an AwaitComplete.\n\n        Args:\n            coroutine: One or more coroutines to execute.\n        '
        self.coroutines = coroutines
        self._future: Future = gather(*self.coroutines)

    async def __call__(self) -> Any:
        return await self

    def __await__(self) -> Iterator[None]:
        if False:
            while True:
                i = 10
        return self._future.__await__()

    @property
    def is_done(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns True if the task has completed.'
        return self._future.done()

    @property
    def exception(self) -> BaseException | None:
        if False:
            i = 10
            return i + 15
        'An exception if it occurred in any of the coroutines.'
        if self._future.done():
            return self._future.exception()
        return None

    @classmethod
    def nothing(cls):
        if False:
            return 10
        'Returns an already completed instance of AwaitComplete.'
        instance = cls()
        instance._future = Future()
        instance._future.set_result(None)
        return instance