import os
import threading
import unittest
from datetime import timedelta
from time import sleep
import pytest
from reactivex.disposable import Disposable
from reactivex.internal.basic import default_now
from reactivex.internal.constants import DELTA_ZERO
from reactivex.internal.exceptions import WouldBlockException
from reactivex.scheduler import ImmediateScheduler
CI = os.getenv('CI') is not None

class TestImmediateScheduler(unittest.TestCase):

    def test_immediate_singleton(self):
        if False:
            return 10
        scheduler = [ImmediateScheduler(), ImmediateScheduler.singleton()]
        assert scheduler[0] is scheduler[1]
        gate = [threading.Semaphore(0), threading.Semaphore(0)]
        scheduler = [None, None]

        def run(idx):
            if False:
                return 10
            scheduler[idx] = ImmediateScheduler()
            gate[idx].release()
        for idx in (0, 1):
            threading.Thread(target=run, args=(idx,)).start()
            gate[idx].acquire()
        assert scheduler[0] is not None
        assert scheduler[1] is not None
        assert scheduler[0] is scheduler[1]

    def test_immediate_extend(self):
        if False:
            i = 10
            return i + 15

        class MyScheduler(ImmediateScheduler):
            pass
        scheduler = [MyScheduler(), MyScheduler.singleton(), ImmediateScheduler.singleton()]
        assert scheduler[0] is scheduler[1]
        assert scheduler[0] is not scheduler[2]

    @pytest.mark.skipif(CI, reason='Flaky test in GitHub Actions')
    def test_immediate_now(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = ImmediateScheduler()
        diff = scheduler.now - default_now()
        assert abs(diff) <= timedelta(milliseconds=1)

    @pytest.mark.skipif(CI, reason='Flaky test in GitHub Actions')
    def test_immediate_now_units(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = ImmediateScheduler()
        diff = scheduler.now
        sleep(1.1)
        diff = scheduler.now - diff
        assert timedelta(milliseconds=1000) < diff < timedelta(milliseconds=1300)

    def test_immediate_scheduleaction(self):
        if False:
            i = 10
            return i + 15
        scheduler = ImmediateScheduler()
        ran = False

        def action(scheduler, state=None):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal ran
            ran = True
        scheduler.schedule(action)
        assert ran

    def test_immediate_schedule_action_error(self):
        if False:
            print('Hello World!')
        scheduler = ImmediateScheduler()

        class MyException(Exception):
            pass

        def action(scheduler, state=None):
            if False:
                return 10
            raise MyException()
        with pytest.raises(MyException):
            return scheduler.schedule(action)

    def test_immediate_schedule_action_due_error(self):
        if False:
            return 10
        scheduler = ImmediateScheduler()
        ran = False

        def action(scheduler, state=None):
            if False:
                i = 10
                return i + 15
            nonlocal ran
            ran = True
        with pytest.raises(WouldBlockException):
            scheduler.schedule_relative(0.1, action)
        assert ran is False

    def test_immediate_simple1(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = ImmediateScheduler()
        xx = 0

        def action(scheduler, state=None):
            if False:
                i = 10
                return i + 15
            nonlocal xx
            xx = state
            return Disposable()
        scheduler.schedule(action, 42)
        assert xx == 42

    def test_immediate_simple2(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = ImmediateScheduler()
        xx = 0

        def action(scheduler, state=None):
            if False:
                i = 10
                return i + 15
            nonlocal xx
            xx = state
            return Disposable()
        scheduler.schedule_absolute(default_now(), action, 42)
        assert xx == 42

    def test_immediate_simple3(self):
        if False:
            return 10
        scheduler = ImmediateScheduler()
        xx = 0

        def action(scheduler, state=None):
            if False:
                print('Hello World!')
            nonlocal xx
            xx = state
            return Disposable()
        scheduler.schedule_relative(DELTA_ZERO, action, 42)
        assert xx == 42

    def test_immediate_recursive1(self):
        if False:
            while True:
                i = 10
        scheduler = ImmediateScheduler()
        xx = 0
        yy = 0

        def action(scheduler, state=None):
            if False:
                i = 10
                return i + 15
            nonlocal xx
            xx = state

            def inner_action(scheduler, state=None):
                if False:
                    print('Hello World!')
                nonlocal yy
                yy = state
                return Disposable()
            return scheduler.schedule(inner_action, 43)
        scheduler.schedule(action, 42)
        assert xx == 42
        assert yy == 43

    def test_immediate_recursive2(self):
        if False:
            print('Hello World!')
        scheduler = ImmediateScheduler()
        xx = 0
        yy = 0

        def action(scheduler, state=None):
            if False:
                while True:
                    i = 10
            nonlocal xx
            xx = state

            def inner_action(scheduler, state=None):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal yy
                yy = state
                return Disposable()
            return scheduler.schedule_absolute(default_now(), inner_action, 43)
        scheduler.schedule_absolute(default_now(), action, 42)
        assert xx == 42
        assert yy == 43

    def test_immediate_recursive3(self):
        if False:
            return 10
        scheduler = ImmediateScheduler()
        xx = 0
        yy = 0

        def action(scheduler, state=None):
            if False:
                i = 10
                return i + 15
            nonlocal xx
            xx = state

            def inner_action(scheduler, state):
                if False:
                    print('Hello World!')
                nonlocal yy
                yy = state
                return Disposable()
            return scheduler.schedule_relative(DELTA_ZERO, inner_action, 43)
        scheduler.schedule_relative(DELTA_ZERO, action, 42)
        assert xx == 42
        assert yy == 43