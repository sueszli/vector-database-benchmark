import asyncio
import os
import threading
import unittest
from datetime import datetime, timedelta
import pytest
from reactivex.scheduler.eventloop import AsyncIOThreadSafeScheduler
CI = os.getenv('CI') is not None

class TestAsyncIOThreadSafeScheduler(unittest.TestCase):

    @pytest.mark.skipif(CI, reason='Flaky test in GitHub Actions')
    def test_asyncio_threadsafe_schedule_now(self):
        if False:
            while True:
                i = 10
        loop = asyncio.get_event_loop()
        scheduler = AsyncIOThreadSafeScheduler(loop)
        diff = scheduler.now - datetime.utcfromtimestamp(loop.time())
        assert abs(diff) < timedelta(milliseconds=2)

    @pytest.mark.skipif(CI, reason='Flaky test in GitHub Actions')
    def test_asyncio_threadsafe_schedule_now_units(self):
        if False:
            return 10
        loop = asyncio.get_event_loop()
        scheduler = AsyncIOThreadSafeScheduler(loop)
        diff = scheduler.now
        yield from asyncio.sleep(0.1)
        diff = scheduler.now - diff
        assert timedelta(milliseconds=80) < diff < timedelta(milliseconds=180)

    def test_asyncio_threadsafe_schedule_action(self):
        if False:
            for i in range(10):
                print('nop')
        loop = asyncio.get_event_loop()

        async def go():
            scheduler = AsyncIOThreadSafeScheduler(loop)
            ran = False

            def action(scheduler, state):
                if False:
                    return 10
                nonlocal ran
                ran = True

            def schedule():
                if False:
                    for i in range(10):
                        print('nop')
                scheduler.schedule(action)
            threading.Thread(target=schedule).start()
            await asyncio.sleep(0.1)
            assert ran is True
        loop.run_until_complete(go())

    def test_asyncio_threadsafe_schedule_action_due(self):
        if False:
            i = 10
            return i + 15
        loop = asyncio.get_event_loop()

        async def go():
            scheduler = AsyncIOThreadSafeScheduler(loop)
            starttime = loop.time()
            endtime = None

            def action(scheduler, state):
                if False:
                    while True:
                        i = 10
                nonlocal endtime
                endtime = loop.time()

            def schedule():
                if False:
                    while True:
                        i = 10
                scheduler.schedule_relative(0.2, action)
            threading.Thread(target=schedule).start()
            await asyncio.sleep(0.3)
            assert endtime is not None
            diff = endtime - starttime
            assert diff > 0.18
        loop.run_until_complete(go())

    def test_asyncio_threadsafe_schedule_action_cancel(self):
        if False:
            i = 10
            return i + 15
        loop = asyncio.get_event_loop()

        async def go():
            ran = False
            scheduler = AsyncIOThreadSafeScheduler(loop)

            def action(scheduler, state):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal ran
                ran = True

            def schedule():
                if False:
                    for i in range(10):
                        print('nop')
                d = scheduler.schedule_relative(0.05, action)
                d.dispose()
            threading.Thread(target=schedule).start()
            await asyncio.sleep(0.3)
            assert ran is False
        loop.run_until_complete(go())

    def cancel_same_thread_common(self, test_body):
        if False:
            i = 10
            return i + 15
        update_state = {'ran': False, 'dispose_completed': False}

        def action(scheduler, state):
            if False:
                return 10
            update_state['ran'] = True

        def thread_target():
            if False:
                print('Hello World!')
            loop = asyncio.new_event_loop()
            scheduler = AsyncIOThreadSafeScheduler(loop)
            test_body(scheduler, action, update_state)

            async def go():
                await asyncio.sleep(0.2)
            loop.run_until_complete(go())
        thread = threading.Thread(target=thread_target)
        thread.daemon = True
        thread.start()
        thread.join(0.3)
        assert update_state['dispose_completed'] is True
        assert update_state['ran'] is False

    def test_asyncio_threadsafe_cancel_non_relative_same_thread(self):
        if False:
            i = 10
            return i + 15

        def test_body(scheduler, action, update_state):
            if False:
                while True:
                    i = 10
            d = scheduler.schedule(action)
            d.dispose()
            update_state['dispose_completed'] = True
        self.cancel_same_thread_common(test_body)

    def test_asyncio_threadsafe_schedule_action_cancel_same_thread(self):
        if False:
            i = 10
            return i + 15

        def test_body(scheduler, action, update_state):
            if False:
                return 10
            d = scheduler.schedule_relative(0.05, action)
            d.dispose()
            update_state['dispose_completed'] = True
        self.cancel_same_thread_common(test_body)

    def test_asyncio_threadsafe_schedule_action_cancel_same_loop(self):
        if False:
            return 10

        def test_body(scheduler, action, update_state):
            if False:
                for i in range(10):
                    print('nop')
            d = scheduler.schedule_relative(0.1, action)

            def do_dispose():
                if False:
                    for i in range(10):
                        print('nop')
                d.dispose()
                update_state['dispose_completed'] = True
            scheduler._loop.call_soon(do_dispose)
        self.cancel_same_thread_common(test_body)