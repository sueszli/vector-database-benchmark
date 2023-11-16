import unittest
from datetime import datetime, timedelta
from time import sleep
import pytest
from reactivex.scheduler.eventloop import IOLoopScheduler
tornado = pytest.importorskip('tornado')
from tornado import ioloop

class TestIOLoopScheduler(unittest.TestCase):

    def test_ioloop_schedule_now(self):
        if False:
            while True:
                i = 10
        loop = ioloop.IOLoop.instance()
        scheduler = IOLoopScheduler(loop)
        diff = scheduler.now - datetime.utcfromtimestamp(loop.time())
        assert abs(diff) < timedelta(milliseconds=1)

    def test_ioloop_schedule_now_units(self):
        if False:
            i = 10
            return i + 15
        loop = ioloop.IOLoop.instance()
        scheduler = IOLoopScheduler(loop)
        diff = scheduler.now
        sleep(0.1)
        diff = scheduler.now - diff
        assert timedelta(milliseconds=80) < diff < timedelta(milliseconds=180)

    def test_ioloop_schedule_action(self):
        if False:
            return 10
        loop = ioloop.IOLoop.instance()
        scheduler = IOLoopScheduler(loop)
        ran = False

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal ran
            ran = True
        scheduler.schedule(action)

        def done():
            if False:
                return 10
            assert ran is True
            loop.stop()
        loop.call_later(0.1, done)
        loop.start()

    def test_ioloop_schedule_action_due(self):
        if False:
            while True:
                i = 10
        loop = ioloop.IOLoop.instance()
        scheduler = IOLoopScheduler(loop)
        starttime = loop.time()
        endtime = None

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal endtime
            endtime = loop.time()
        scheduler.schedule_relative(0.2, action)

        def done():
            if False:
                for i in range(10):
                    print('nop')
            assert endtime is not None
            diff = endtime - starttime
            assert diff > 0.18
            loop.stop()
        loop.call_later(0.3, done)
        loop.start()

    def test_ioloop_schedule_action_cancel(self):
        if False:
            return 10
        loop = ioloop.IOLoop.instance()
        ran = False
        scheduler = IOLoopScheduler(loop)

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal ran
            ran = True
        d = scheduler.schedule_relative(0.01, action)
        d.dispose()

        def done():
            if False:
                return 10
            assert ran is False
            loop.stop()
        loop.call_later(0.1, done)
        loop.start()