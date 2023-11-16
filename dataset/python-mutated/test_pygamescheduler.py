import unittest
from datetime import timedelta
from time import sleep
import pytest
from reactivex.internal.basic import default_now
from reactivex.scheduler.mainloop import PyGameScheduler
pygame = pytest.importorskip('pygame')

class TestPyGameScheduler(unittest.TestCase):

    def test_pygame_schedule_now(self):
        if False:
            print('Hello World!')
        scheduler = PyGameScheduler(pygame)
        diff = scheduler.now - default_now()
        assert abs(diff) < timedelta(milliseconds=1)

    def test_pygame_schedule_now_units(self):
        if False:
            i = 10
            return i + 15
        scheduler = PyGameScheduler(pygame)
        diff = scheduler.now
        sleep(0.1)
        diff = scheduler.now - diff
        assert timedelta(milliseconds=80) < diff < timedelta(milliseconds=180)

    def test_pygame_schedule_action(self):
        if False:
            print('Hello World!')
        scheduler = PyGameScheduler(pygame)
        ran = False

        def action(scheduler, state):
            if False:
                while True:
                    i = 10
            nonlocal ran
            ran = True
        scheduler.schedule(action)
        scheduler.run()
        assert ran is True

    def test_pygame_schedule_action_due_relative(self):
        if False:
            while True:
                i = 10
        scheduler = PyGameScheduler(pygame)
        starttime = default_now()
        endtime = None

        def action(scheduler, state):
            if False:
                return 10
            nonlocal endtime
            endtime = default_now()
        scheduler.schedule_relative(0.1, action)
        scheduler.run()
        assert endtime is None
        sleep(0.2)
        scheduler.run()
        assert endtime is not None
        diff = endtime - starttime
        assert diff > timedelta(milliseconds=180)

    def test_pygame_schedule_action_due_absolute(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = PyGameScheduler(pygame)
        starttime = default_now()
        endtime = None

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal endtime
            endtime = default_now()
        scheduler.schedule_absolute(starttime + timedelta(seconds=0.1), action)
        scheduler.run()
        assert endtime is None
        sleep(0.2)
        scheduler.run()
        assert endtime is not None
        diff = endtime - starttime
        assert diff > timedelta(milliseconds=180)

    def test_pygame_schedule_action_cancel(self):
        if False:
            return 10
        scheduler = PyGameScheduler(pygame)
        ran = False

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal ran
            ran = True
        d = scheduler.schedule_relative(0.1, action)
        d.dispose()
        sleep(0.2)
        scheduler.run()
        assert ran is False