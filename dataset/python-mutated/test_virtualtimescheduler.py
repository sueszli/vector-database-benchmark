import unittest
import pytest
from reactivex.internal import ArgumentOutOfRangeException
from reactivex.internal.constants import DELTA_ZERO, UTC_ZERO
from reactivex.scheduler import VirtualTimeScheduler

class VirtualSchedulerTestScheduler(VirtualTimeScheduler):

    def add(self, absolute, relative):
        if False:
            return 10
        return absolute + relative

class TestVirtualTimeScheduler(unittest.TestCase):

    def test_virtual_now_noarg(self):
        if False:
            print('Hello World!')
        scheduler = VirtualSchedulerTestScheduler()
        assert scheduler.clock == 0.0
        assert scheduler.now == UTC_ZERO

    def test_virtual_now_float(self):
        if False:
            return 10
        scheduler = VirtualSchedulerTestScheduler(0.0)
        assert scheduler.clock == 0.0
        assert scheduler.now == UTC_ZERO

    def test_virtual_now_timedelta(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = VirtualSchedulerTestScheduler(DELTA_ZERO)
        assert scheduler.clock == DELTA_ZERO
        assert scheduler.now == UTC_ZERO

    def test_virtual_now_datetime(self):
        if False:
            while True:
                i = 10
        scheduler = VirtualSchedulerTestScheduler(UTC_ZERO)
        assert scheduler.clock == UTC_ZERO
        assert scheduler.now == UTC_ZERO

    def test_virtual_schedule_action(self):
        if False:
            print('Hello World!')
        scheduler = VirtualSchedulerTestScheduler()
        ran = False

        def action(scheduler, state):
            if False:
                i = 10
                return i + 15
            nonlocal ran
            ran = True
        scheduler.schedule(action)
        scheduler.start()
        assert ran is True

    def test_virtual_schedule_action_error(self):
        if False:
            return 10
        scheduler = VirtualSchedulerTestScheduler()

        class MyException(Exception):
            pass

        def action(scheduler, state):
            if False:
                return 10
            raise MyException()
        with pytest.raises(MyException):
            scheduler.schedule(action)
            scheduler.start()

    def test_virtual_schedule_sleep_error(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = VirtualSchedulerTestScheduler()
        with pytest.raises(ArgumentOutOfRangeException):
            scheduler.sleep(-1)

    def test_virtual_schedule_advance_clock_error(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = VirtualSchedulerTestScheduler()
        with pytest.raises(ArgumentOutOfRangeException):
            scheduler.advance_to(scheduler._clock - 1)