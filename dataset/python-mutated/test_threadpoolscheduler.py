import os
import threading
import unittest
from datetime import timedelta
from time import sleep
import pytest
from reactivex.internal.basic import default_now
from reactivex.scheduler import ThreadPoolScheduler
thread_pool_scheduler = ThreadPoolScheduler()
CI = os.getenv('CI') is not None

class TestThreadPoolScheduler(unittest.TestCase):

    @pytest.mark.skipif(CI, reason='Flaky test in GitHub Actions')
    def test_threadpool_now(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = ThreadPoolScheduler()
        diff = scheduler.now - default_now()
        assert abs(diff) < timedelta(milliseconds=5)

    def test_threadpool_now_units(self):
        if False:
            i = 10
            return i + 15
        scheduler = ThreadPoolScheduler()
        diff = scheduler.now
        sleep(1.1)
        diff = scheduler.now - diff
        assert timedelta(milliseconds=1000) < diff < timedelta(milliseconds=1300)

    def test_schedule_action(self):
        if False:
            print('Hello World!')
        ident = threading.current_thread().ident
        evt = threading.Event()
        nt = thread_pool_scheduler

        def action(scheduler, state):
            if False:
                return 10
            assert ident != threading.current_thread().ident
            evt.set()
        nt.schedule(action)
        evt.wait()

    def test_schedule_action_due_relative(self):
        if False:
            for i in range(10):
                print('nop')
        ident = threading.current_thread().ident
        evt = threading.Event()
        nt = thread_pool_scheduler

        def action(scheduler, state):
            if False:
                i = 10
                return i + 15
            assert ident != threading.current_thread().ident
            evt.set()
        nt.schedule_relative(timedelta(milliseconds=200), action)
        evt.wait()

    def test_schedule_action_due_0(self):
        if False:
            return 10
        ident = threading.current_thread().ident
        evt = threading.Event()
        nt = thread_pool_scheduler

        def action(scheduler, state):
            if False:
                return 10
            assert ident != threading.current_thread().ident
            evt.set()
        nt.schedule_relative(0.1, action)
        evt.wait()

    def test_schedule_action_absolute(self):
        if False:
            i = 10
            return i + 15
        ident = threading.current_thread().ident
        evt = threading.Event()
        nt = thread_pool_scheduler

        def action(scheduler, state):
            if False:
                while True:
                    i = 10
            assert ident != threading.current_thread().ident
            evt.set()
        nt.schedule_absolute(default_now() + timedelta(milliseconds=100), action)
        evt.wait()

    def test_schedule_action_cancel(self):
        if False:
            return 10
        nt = thread_pool_scheduler
        ran = False

        def action(scheduler, state):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal ran
            ran = True
        d = nt.schedule_relative(0.05, action)
        d.dispose()
        sleep(0.1)
        assert ran is False