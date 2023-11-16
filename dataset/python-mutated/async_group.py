from __future__ import annotations
import asyncio
import logging
from asyncio import CancelledError, Future, Task
from contextlib import suppress
from typing import Coroutine, Iterable, List, Optional, Set
from weakref import ref
from tribler.core.utilities.async_group.exceptions import DoneException

def done_callback(group_ref):
    if False:
        while True:
            i = 10

    def actual_callback(future):
        if False:
            return 10
        AsyncGroup.global_futures.discard(future)
        group: Optional[AsyncGroup] = group_ref()
        if group is not None:
            group.futures.discard(future)
    return actual_callback

class AsyncGroup:
    """This class is a little brother of TaskManager and its purpose is only to cancel or to wait a group
    of asyncio Tasks/Futures.

    Example:
    >>> from tribler.core.utilities.async_group.async_group import AsyncGroup
    >>> async def void():
    ...     pass
    >>> group = AsyncGroup()
    >>> group.add_task(void())
    >>> group.add_task(void())
    >>> group.add_task(void())
    >>> await group.cancel()
    """
    global_futures: Set[Future] = set()

    def __init__(self):
        if False:
            while True:
                i = 10
        self._logger = logging.getLogger(self.__class__.__name__)
        self.ref = ref(self)
        self.futures: Set[Future] = set()
        self._done = False

    def add_task(self, coroutine: Coroutine) -> Task:
        if False:
            i = 10
            return i + 15
        'Add a coroutine to the group.\n        '
        task = asyncio.create_task(coroutine)
        if self._done:
            task.cancel()
            raise DoneException()
        self.futures.add(task)
        self.global_futures.add(task)
        task.add_done_callback(done_callback(self.ref))
        return task

    async def wait(self):
        """ Wait for completion of all futures
        """
        while (active := set(self._active(self.futures))):
            await asyncio.wait(active)
        self._done = True

    async def cancel(self) -> List[Future]:
        """Cancel the group.

        Only active futures will be cancelled.
        """
        if self._done:
            return []
        self._done = True
        active = list(self._active(self.futures))
        for future in active:
            future.cancel()
        with suppress(CancelledError):
            await asyncio.gather(*active)
        return active

    @property
    def done(self):
        if False:
            print('Hello World!')
        return self._done

    @staticmethod
    def _active(futures: Iterable[Future]) -> Iterable[Future]:
        if False:
            print('Hello World!')
        return (future for future in futures if not future.done())

    def __del__(self):
        if False:
            i = 10
            return i + 15
        if (active := list(self._active(self.futures))):
            self._logger.error(f'AsyncGroup is destroying but {len(active)} futures are active')
            for future in active:
                future.cancel()