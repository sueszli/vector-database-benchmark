import asyncio
import unittest
from reactivex import operators as ops
from reactivex.scheduler.eventloop import AsyncIOScheduler
from reactivex.subject import Subject

class TestFlatMapAsync(unittest.TestCase):

    def test_flat_map_async(self):
        if False:
            return 10
        actual_next = None
        loop = asyncio.get_event_loop()
        scheduler = AsyncIOScheduler(loop=loop)

        def mapper(i: int):
            if False:
                while True:
                    i = 10

            async def _mapper(i: int):
                return i + 1
            return asyncio.ensure_future(_mapper(i))

        def on_next(i: int):
            if False:
                while True:
                    i = 10
            nonlocal actual_next
            actual_next = i

        def on_error(ex):
            if False:
                i = 10
                return i + 15
            print('Error', ex)

        async def test_flat_map():
            x: Subject[int] = Subject()
            x.pipe(ops.flat_map(mapper)).subscribe(on_next, on_error, scheduler=scheduler)
            x.on_next(10)
            await asyncio.sleep(0.1)
        loop.run_until_complete(test_flat_map())
        assert actual_next == 11