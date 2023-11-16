import functools
import sys
import typing as t
from asyncio import sleep
from tenacity import AttemptManager
from tenacity import BaseRetrying
from tenacity import DoAttempt
from tenacity import DoSleep
from tenacity import RetryCallState
WrappedFnReturnT = t.TypeVar('WrappedFnReturnT')
WrappedFn = t.TypeVar('WrappedFn', bound=t.Callable[..., t.Awaitable[t.Any]])

class AsyncRetrying(BaseRetrying):
    sleep: t.Callable[[float], t.Awaitable[t.Any]]

    def __init__(self, sleep: t.Callable[[float], t.Awaitable[t.Any]]=sleep, **kwargs: t.Any) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.sleep = sleep

    async def __call__(self, fn: WrappedFn, *args: t.Any, **kwargs: t.Any) -> WrappedFnReturnT:
        self.begin()
        retry_state = RetryCallState(retry_object=self, fn=fn, args=args, kwargs=kwargs)
        while True:
            do = self.iter(retry_state=retry_state)
            if isinstance(do, DoAttempt):
                try:
                    result = await fn(*args, **kwargs)
                except BaseException:
                    retry_state.set_exception(sys.exc_info())
                else:
                    retry_state.set_result(result)
            elif isinstance(do, DoSleep):
                retry_state.prepare_for_next_attempt()
                await self.sleep(do)
            else:
                return do

    def __iter__(self) -> t.Generator[AttemptManager, None, None]:
        if False:
            return 10
        raise TypeError('AsyncRetrying object is not iterable')

    def __aiter__(self) -> 'AsyncRetrying':
        if False:
            while True:
                i = 10
        self.begin()
        self._retry_state = RetryCallState(self, fn=None, args=(), kwargs={})
        return self

    async def __anext__(self) -> AttemptManager:
        while True:
            do = self.iter(retry_state=self._retry_state)
            if do is None:
                raise StopAsyncIteration
            elif isinstance(do, DoAttempt):
                return AttemptManager(retry_state=self._retry_state)
            elif isinstance(do, DoSleep):
                self._retry_state.prepare_for_next_attempt()
                await self.sleep(do)
            else:
                raise StopAsyncIteration

    def wraps(self, fn: WrappedFn) -> WrappedFn:
        if False:
            return 10
        fn = super().wraps(fn)

        @functools.wraps(fn, functools.WRAPPER_ASSIGNMENTS + ('__defaults__', '__kwdefaults__'))
        async def async_wrapped(*args: t.Any, **kwargs: t.Any) -> t.Any:
            return await fn(*args, **kwargs)
        async_wrapped.retry = fn.retry
        async_wrapped.retry_with = fn.retry_with
        return async_wrapped