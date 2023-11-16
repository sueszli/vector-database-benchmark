import asyncio
from asyncio import Future
import reactivex
from reactivex import Observable
from reactivex import operators as ops
from reactivex.scheduler.eventloop import AsyncIOScheduler

def to_async_iterable():
    if False:
        for i in range(10):
            print('nop')

    def _to_async_iterable(source: Observable):
        if False:
            for i in range(10):
                print('nop')

        class AIterable:

            def __aiter__(self):
                if False:
                    while True:
                        i = 10

                class AIterator:

                    def __init__(self):
                        if False:
                            i = 10
                            return i + 15
                        self.notifications = []
                        self.future = Future()
                        source.pipe(ops.materialize()).subscribe(self.on_next)

                    def feeder(self):
                        if False:
                            return 10
                        if not self.notifications or self.future.done():
                            return
                        notification = self.notifications.pop(0)
                        dispatch = {'N': lambda : self.future.set_result(notification.value), 'E': lambda : self.future.set_exception(notification.exception), 'C': lambda : self.future.set_exception(StopAsyncIteration)}
                        dispatch[notification.kind]()

                    def on_next(self, notification):
                        if False:
                            print('Hello World!')
                        self.notifications.append(notification)
                        self.feeder()

                    async def __anext__(self):
                        self.feeder()
                        value = await self.future
                        self.future = Future()
                        return value
                return AIterator()
        return AIterable()
    return _to_async_iterable

async def go(loop):
    scheduler = AsyncIOScheduler(loop)
    ai = reactivex.range(0, 10, scheduler=scheduler).pipe(to_async_iterable())
    async for x in ai:
        print('got %s' % x)

def main():
    if False:
        while True:
            i = 10
    loop = asyncio.get_event_loop()
    loop.run_until_complete(go(loop))
if __name__ == '__main__':
    main()