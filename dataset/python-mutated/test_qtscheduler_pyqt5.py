import threading
from datetime import timedelta
from time import sleep
import pytest
from reactivex.internal.basic import default_now
from reactivex.scheduler.mainloop import QtScheduler
QtCore = pytest.importorskip('PyQt5.QtCore')

@pytest.fixture(scope='module')
def app():
    if False:
        while True:
            i = 10
    app = QtCore.QCoreApplication([])
    yield app

class TestQtSchedulerPyQt5:

    def test_pyqt5_schedule_now(self):
        if False:
            i = 10
            return i + 15
        scheduler = QtScheduler(QtCore)
        diff = scheduler.now - default_now()
        assert abs(diff) < timedelta(milliseconds=1)

    def test_pyqt5_schedule_now_units(self):
        if False:
            i = 10
            return i + 15
        scheduler = QtScheduler(QtCore)
        diff = scheduler.now
        sleep(0.1)
        diff = scheduler.now - diff
        assert timedelta(milliseconds=80) < diff < timedelta(milliseconds=180)

    def test_pyqt5_schedule_action(self, app):
        if False:
            print('Hello World!')
        scheduler = QtScheduler(QtCore)
        gate = threading.Semaphore(0)
        ran = False

        def action(scheduler, state):
            if False:
                while True:
                    i = 10
            nonlocal ran
            ran = True
        scheduler.schedule(action)

        def done():
            if False:
                print('Hello World!')
            app.quit()
            gate.release()
        QtCore.QTimer.singleShot(50, done)
        app.exec_()
        gate.acquire()
        assert ran is True

    def test_pyqt5_schedule_action_due_relative(self, app):
        if False:
            while True:
                i = 10
        scheduler = QtScheduler(QtCore)
        gate = threading.Semaphore(0)
        starttime = default_now()
        endtime = None

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal endtime
            endtime = default_now()
        scheduler.schedule_relative(0.2, action)

        def done():
            if False:
                while True:
                    i = 10
            app.quit()
            gate.release()
        QtCore.QTimer.singleShot(300, done)
        app.exec_()
        gate.acquire()
        assert endtime is not None
        diff = endtime - starttime
        assert diff > timedelta(milliseconds=180)

    def test_pyqt5_schedule_action_due_absolute(self, app):
        if False:
            return 10
        scheduler = QtScheduler(QtCore)
        gate = threading.Semaphore(0)
        starttime = default_now()
        endtime = None

        def action(scheduler, state):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal endtime
            endtime = default_now()
        scheduler.schedule_absolute(starttime + timedelta(seconds=0.2), action)

        def done():
            if False:
                i = 10
                return i + 15
            app.quit()
            gate.release()
        QtCore.QTimer.singleShot(300, done)
        app.exec_()
        gate.acquire()
        assert endtime is not None
        diff = endtime - starttime
        assert diff > timedelta(milliseconds=180)

    def test_pyqt5_schedule_action_cancel(self, app):
        if False:
            while True:
                i = 10
        ran = False
        scheduler = QtScheduler(QtCore)
        gate = threading.Semaphore(0)

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal ran
            ran = True
        d = scheduler.schedule_relative(0.1, action)
        d.dispose()

        def done():
            if False:
                for i in range(10):
                    print('nop')
            app.quit()
            gate.release()
        QtCore.QTimer.singleShot(300, done)
        app.exec_()
        gate.acquire()
        assert ran is False

    def test_pyqt5_schedule_action_periodic(self, app):
        if False:
            for i in range(10):
                print('nop')
        scheduler = QtScheduler(QtCore)
        gate = threading.Semaphore(0)
        period = 0.05
        counter = 3

        def action(state):
            if False:
                print('Hello World!')
            nonlocal counter
            if state:
                counter -= 1
                return state - 1
        scheduler.schedule_periodic(period, action, counter)

        def done():
            if False:
                return 10
            app.quit()
            gate.release()
        QtCore.QTimer.singleShot(300, done)
        app.exec_()
        gate.acquire()
        assert counter == 0

    def test_pyqt5_schedule_periodic_cancel(self, app):
        if False:
            print('Hello World!')
        scheduler = QtScheduler(QtCore)
        gate = threading.Semaphore(0)
        period = 0.05
        counter = 3

        def action(state):
            if False:
                i = 10
                return i + 15
            nonlocal counter
            if state:
                counter -= 1
                return state - 1
        disp = scheduler.schedule_periodic(period, action, counter)

        def dispose():
            if False:
                return 10
            disp.dispose()
        QtCore.QTimer.singleShot(100, dispose)

        def done():
            if False:
                i = 10
                return i + 15
            app.quit()
            gate.release()
        QtCore.QTimer.singleShot(300, done)
        app.exec_()
        gate.acquire()
        assert 0 < counter < 3