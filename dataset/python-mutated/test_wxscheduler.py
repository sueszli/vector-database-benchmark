import unittest
from datetime import timedelta
from time import sleep
import pytest
from reactivex.internal.basic import default_now
from reactivex.scheduler.mainloop import WxScheduler
wx = pytest.importorskip('wx')

def make_app():
    if False:
        print('Hello World!')
    app = wx.App()
    wx.Frame(None)
    return app

class AppExit(wx.Timer):

    def __init__(self, app) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.app = app

    def Notify(self):
        if False:
            return 10
        self.app.ExitMainLoop()

class TestWxScheduler(unittest.TestCase):

    def test_wx_schedule_now(self):
        if False:
            while True:
                i = 10
        scheduler = WxScheduler(wx)
        diff = scheduler.now - default_now()
        assert abs(diff) < timedelta(milliseconds=1)

    def test_wx_schedule_now_units(self):
        if False:
            i = 10
            return i + 15
        scheduler = WxScheduler(wx)
        diff = scheduler.now
        sleep(0.1)
        diff = scheduler.now - diff
        assert timedelta(milliseconds=80) < diff < timedelta(milliseconds=180)

    def test_wx_schedule_action(self):
        if False:
            i = 10
            return i + 15
        app = make_app()
        exit = AppExit(app)
        scheduler = WxScheduler(wx)
        ran = False

        def action(scheduler, state):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal ran
            ran = True
        scheduler.schedule(action)
        exit.Start(100, wx.TIMER_ONE_SHOT)
        app.MainLoop()
        scheduler.cancel_all()
        assert ran is True

    def test_wx_schedule_action_relative(self):
        if False:
            i = 10
            return i + 15
        app = make_app()
        exit = AppExit(app)
        scheduler = WxScheduler(wx)
        starttime = default_now()
        endtime = None

        def action(scheduler, state):
            if False:
                i = 10
                return i + 15
            nonlocal endtime
            endtime = default_now()
        scheduler.schedule_relative(0.1, action)
        exit.Start(200, wx.TIMER_ONE_SHOT)
        app.MainLoop()
        scheduler.cancel_all()
        assert endtime is not None
        diff = endtime - starttime
        assert timedelta(milliseconds=80) < diff < timedelta(milliseconds=180)

    def test_wx_schedule_action_absolute(self):
        if False:
            print('Hello World!')
        app = make_app()
        exit = AppExit(app)
        scheduler = WxScheduler(wx)
        starttime = default_now()
        endtime = None

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal endtime
            endtime = default_now()
        due = scheduler.now + timedelta(milliseconds=100)
        scheduler.schedule_absolute(due, action)
        exit.Start(200, wx.TIMER_ONE_SHOT)
        app.MainLoop()
        scheduler.cancel_all()
        assert endtime is not None
        diff = endtime - starttime
        assert timedelta(milliseconds=80) < diff < timedelta(milliseconds=180)

    def test_wx_schedule_action_cancel(self):
        if False:
            return 10
        app = make_app()
        exit = AppExit(app)
        scheduler = WxScheduler(wx)
        ran = False

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal ran
            ran = True
        d = scheduler.schedule_relative(0.1, action)
        d.dispose()
        exit.Start(200, wx.TIMER_ONE_SHOT)
        app.MainLoop()
        scheduler.cancel_all()
        assert ran is False

    def test_wx_schedule_action_periodic(self):
        if False:
            print('Hello World!')
        app = make_app()
        exit = AppExit(app)
        scheduler = WxScheduler(wx)
        period = 0.05
        counter = 3

        def action(state):
            if False:
                while True:
                    i = 10
            nonlocal counter
            if state:
                counter -= 1
                return state - 1
        scheduler.schedule_periodic(period, action, counter)
        exit.Start(500, wx.TIMER_ONE_SHOT)
        app.MainLoop()
        scheduler.cancel_all()
        assert counter == 0