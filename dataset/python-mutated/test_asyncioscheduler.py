import asyncio
import os
import unittest
from datetime import datetime, timedelta
import pytest
from reactivex.scheduler.eventloop import AsyncIOScheduler
CI = os.getenv('CI') is not None

class TestAsyncIOScheduler(unittest.TestCase):

    @pytest.mark.skipif(CI, reason='Flaky test in GitHub Actions')
    def test_asyncio_schedule_now(self):
        if False:
            print('Hello World!')
        loop = asyncio.get_event_loop()
        scheduler = AsyncIOScheduler(loop)
        diff = scheduler.now - datetime.utcfromtimestamp(loop.time())
        assert abs(diff) < timedelta(milliseconds=2)

    @pytest.mark.skipif(CI, reason='Test is flaky in GitHub Actions')
    def test_asyncio_schedule_now_units(self):
        if False:
            return 10
        loop = asyncio.get_event_loop()
        scheduler = AsyncIOScheduler(loop)
        diff = scheduler.now
        yield from asyncio.sleep(0.1)
        diff = scheduler.now - diff
        assert timedelta(milliseconds=80) < diff < timedelta(milliseconds=180)

    def test_asyncio_schedule_action(self):
        if False:
            return 10
        loop = asyncio.get_event_loop()

        async def go():
            scheduler = AsyncIOScheduler(loop)
            ran = False

            def action(scheduler, state):
                if False:
                    i = 10
                    return i + 15
                nonlocal ran
                ran = True
            scheduler.schedule(action)
            await asyncio.sleep(0.1)
            assert ran is True
        loop.run_until_complete(go())

    def test_asyncio_schedule_action_due(self):
        if False:
            while True:
                i = 10
        loop = asyncio.get_event_loop()

        async def go():
            scheduler = AsyncIOScheduler(loop)
            starttime = loop.time()
            endtime = None

            def action(scheduler, state):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal endtime
                endtime = loop.time()
            scheduler.schedule_relative(0.2, action)
            await asyncio.sleep(0.3)
            assert endtime is not None
            diff = endtime - starttime
            assert diff > 0.18
        loop.run_until_complete(go())

    def test_asyncio_schedule_action_cancel(self):
        if False:
            while True:
                i = 10
        loop = asyncio.get_event_loop()

        async def go():
            ran = False
            scheduler = AsyncIOScheduler(loop)

            def action(scheduler, state):
                if False:
                    i = 10
                    return i + 15
                nonlocal ran
                ran = True
            d = scheduler.schedule_relative(0.05, action)
            d.dispose()
            await asyncio.sleep(0.3)
            assert ran is False
        loop.run_until_complete(go())