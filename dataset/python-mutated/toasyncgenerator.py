import asyncio
from asyncio import Future
from typing import Any, Coroutine, List, TypeVar
import reactivex
from reactivex import Notification, Observable
from reactivex import operators as ops
from reactivex.scheduler.eventloop import AsyncIOScheduler
_T = TypeVar('_T')

def to_async_generator(sentinel: Any=None) -> Coroutine[Any, Any, Future[Any]]:
    if False:
        print('Hello World!')
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    notifications: List[Notification[Any]] = []

    def _to_async_generator(source: Observable[_T]):
        if False:
            print('Hello World!')

        def feeder():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal future
            if not notifications or future.done():
                return
            notification = notifications.pop(0)
            if notification.kind == 'E':
                future.set_exception(notification.exception)
            elif notification.kind == 'C':
                future.set_result(sentinel)
            else:
                future.set_result(notification.value)

        def on_next(value: _T) -> None:
            if False:
                while True:
                    i = 10
            'Takes on_next values and appends them to the notification queue'
            notifications.append(value)
            loop.call_soon(feeder)
        source.pipe(ops.materialize()).subscribe(on_next)

        async def gen():
            """Generator producing futures"""
            nonlocal future
            loop.call_soon(feeder)
            future = Future()
            return future
        return gen
    return _to_async_generator

async def go(loop):
    scheduler = AsyncIOScheduler(loop)
    xs = reactivex.from_([x for x in range(10)], scheduler=scheduler)
    gen = xs.pipe(to_async_generator())
    while True:
        x = await gen()
        if x is None:
            break
        print(x)

def main():
    if False:
        for i in range(10):
            print('nop')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(go(loop))
if __name__ == '__main__':
    main()