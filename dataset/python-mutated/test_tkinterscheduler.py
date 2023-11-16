import os
import unittest
from datetime import timedelta
from time import sleep
import pytest
from reactivex.internal.basic import default_now
from reactivex.scheduler.mainloop import TkinterScheduler
tkinter = pytest.importorskip('tkinter')
try:
    root = tkinter.Tk()
    root.withdraw()
    display = True
except Exception:
    display = False
CI = os.getenv('CI') is not None

@pytest.mark.skipif('display == False')
class TestTkinterScheduler(unittest.TestCase):

    @pytest.mark.skipif(CI, reason='Flaky test in GitHub Actions')
    def test_tkinter_schedule_now(self):
        if False:
            while True:
                i = 10
        scheduler = TkinterScheduler(root)
        res = scheduler.now - default_now()
        assert abs(res) <= timedelta(milliseconds=1)

    @pytest.mark.skipif(CI, reason='Flaky test in GitHub Actions')
    def test_tkinter_schedule_now_units(self):
        if False:
            print('Hello World!')
        scheduler = TkinterScheduler(root)
        diff = scheduler.now
        sleep(1.1)
        diff = scheduler.now - diff
        assert timedelta(milliseconds=1000) < diff < timedelta(milliseconds=1300)

    def test_tkinter_schedule_action(self):
        if False:
            i = 10
            return i + 15
        scheduler = TkinterScheduler(root)
        ran = False

        def action(scheduler, state):
            if False:
                return 10
            nonlocal ran
            ran = True
        scheduler.schedule(action)

        def done():
            if False:
                print('Hello World!')
            assert ran is True
            root.quit()
        root.after_idle(done)
        root.mainloop()

    def test_tkinter_schedule_action_due(self):
        if False:
            i = 10
            return i + 15
        scheduler = TkinterScheduler(root)
        starttime = default_now()
        endtime = None

        def action(scheduler, state):
            if False:
                return 10
            nonlocal endtime
            endtime = default_now()
        scheduler.schedule_relative(0.2, action)

        def done():
            if False:
                print('Hello World!')
            root.quit()
            assert endtime is not None
            diff = endtime - starttime
            assert diff > timedelta(milliseconds=180)
        root.after(300, done)
        root.mainloop()

    def test_tkinter_schedule_action_cancel(self):
        if False:
            print('Hello World!')
        ran = False
        scheduler = TkinterScheduler(root)

        def action(scheduler, state):
            if False:
                while True:
                    i = 10
            nonlocal ran
            ran = True
        d = scheduler.schedule_relative(0.1, action)
        d.dispose()

        def done():
            if False:
                print('Hello World!')
            root.quit()
            assert ran is False
        root.after(300, done)
        root.mainloop()

    def test_tkinter_schedule_action_periodic(self):
        if False:
            i = 10
            return i + 15
        scheduler = TkinterScheduler(root)
        period = 0.05
        counter = 3

        def action(state):
            if False:
                return 10
            nonlocal counter
            if state:
                counter -= 1
                return state - 1
        scheduler.schedule_periodic(period, action, counter)

        def done():
            if False:
                while True:
                    i = 10
            root.quit()
            assert counter == 0
        root.after(300, done)
        root.mainloop()