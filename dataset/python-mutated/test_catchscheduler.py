import unittest
from datetime import timedelta
from reactivex.scheduler import CatchScheduler, VirtualTimeScheduler

class MyException(Exception):
    pass

class CatchSchedulerTestScheduler(VirtualTimeScheduler):

    def __init__(self, initial_clock=0.0):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(initial_clock)
        self.exc = None

    def add(self, absolute, relative):
        if False:
            i = 10
            return i + 15
        return absolute + relative

    def _wrap(self, action):
        if False:
            print('Hello World!')

        def _action(scheduler, state=None):
            if False:
                while True:
                    i = 10
            ret = None
            try:
                ret = action(scheduler, state)
            except MyException as e:
                self.exc = e
            finally:
                return ret
        return _action

    def schedule_absolute(self, duetime, action, state=None):
        if False:
            i = 10
            return i + 15
        action = self._wrap(action)
        return super().schedule_absolute(duetime, action, state=state)

class TestCatchScheduler(unittest.TestCase):

    def test_catch_now(self):
        if False:
            for i in range(10):
                print('nop')
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, lambda ex: True)
        diff = scheduler.now - wrapped.now
        assert abs(diff) < timedelta(milliseconds=1)

    def test_catch_now_units(self):
        if False:
            print('Hello World!')
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, lambda ex: True)
        diff = scheduler.now
        wrapped.sleep(0.1)
        diff = scheduler.now - diff
        assert timedelta(milliseconds=80) < diff < timedelta(milliseconds=180)

    def test_catch_schedule(self):
        if False:
            while True:
                i = 10
        ran = False
        handled = False

        def action(scheduler, state):
            if False:
                return 10
            nonlocal ran
            ran = True

        def handler(_):
            if False:
                i = 10
                return i + 15
            nonlocal handled
            handled = True
            return True
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, handler)
        scheduler.schedule(action)
        wrapped.start()
        assert ran is True
        assert handled is False
        assert wrapped.exc is None

    def test_catch_schedule_relative(self):
        if False:
            while True:
                i = 10
        ran = False
        handled = False

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal ran
            ran = True

        def handler(_):
            if False:
                while True:
                    i = 10
            nonlocal handled
            handled = True
            return True
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, handler)
        scheduler.schedule_relative(0.1, action)
        wrapped.start()
        assert ran is True
        assert handled is False
        assert wrapped.exc is None

    def test_catch_schedule_absolute(self):
        if False:
            while True:
                i = 10
        ran = False
        handled = False

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal ran
            ran = True

        def handler(_):
            if False:
                i = 10
                return i + 15
            nonlocal handled
            handled = True
            return True
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, handler)
        scheduler.schedule_absolute(0.1, action)
        wrapped.start()
        assert ran is True
        assert handled is False
        assert wrapped.exc is None

    def test_catch_schedule_error_handled(self):
        if False:
            i = 10
            return i + 15
        ran = False
        handled = False

        def action(scheduler, state):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal ran
            ran = True
            raise MyException()

        def handler(_):
            if False:
                i = 10
                return i + 15
            nonlocal handled
            handled = True
            return True
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, handler)
        scheduler.schedule(action)
        wrapped.start()
        assert ran is True
        assert handled is True
        assert wrapped.exc is None

    def test_catch_schedule_error_unhandled(self):
        if False:
            for i in range(10):
                print('nop')
        ran = False
        handled = False

        def action(scheduler, state):
            if False:
                return 10
            nonlocal ran
            ran = True
            raise MyException()

        def handler(_):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal handled
            handled = True
            return False
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, handler)
        scheduler.schedule(action)
        wrapped.start()
        assert ran is True
        assert handled is True
        assert isinstance(wrapped.exc, MyException)

    def test_catch_schedule_nested(self):
        if False:
            for i in range(10):
                print('nop')
        ran = False
        handled = False

        def inner(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal ran
            ran = True

        def outer(scheduler, state):
            if False:
                print('Hello World!')
            scheduler.schedule(inner)

        def handler(_):
            if False:
                return 10
            nonlocal handled
            handled = True
            return True
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, handler)
        scheduler.schedule(outer)
        wrapped.start()
        assert ran is True
        assert handled is False
        assert wrapped.exc is None

    def test_catch_schedule_nested_error_handled(self):
        if False:
            return 10
        ran = False
        handled = False

        def inner(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal ran
            ran = True
            raise MyException()

        def outer(scheduler, state):
            if False:
                return 10
            scheduler.schedule(inner)

        def handler(_):
            if False:
                print('Hello World!')
            nonlocal handled
            handled = True
            return True
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, handler)
        scheduler.schedule(outer)
        wrapped.start()
        assert ran is True
        assert handled is True
        assert wrapped.exc is None

    def test_catch_schedule_nested_error_unhandled(self):
        if False:
            print('Hello World!')
        ran = False
        handled = False

        def inner(scheduler, state):
            if False:
                while True:
                    i = 10
            nonlocal ran
            ran = True
            raise MyException()

        def outer(scheduler, state):
            if False:
                i = 10
                return i + 15
            scheduler.schedule(inner)

        def handler(_):
            if False:
                return 10
            nonlocal handled
            handled = True
            return False
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, handler)
        scheduler.schedule(outer)
        wrapped.start()
        assert ran is True
        assert handled is True
        assert isinstance(wrapped.exc, MyException)

    def test_catch_schedule_periodic(self):
        if False:
            print('Hello World!')
        period = 0.05
        counter = 3
        handled = False

        def action(state):
            if False:
                while True:
                    i = 10
            nonlocal counter
            if state:
                counter -= 1
                return state - 1
            if counter == 0:
                disp.dispose()

        def handler(_):
            if False:
                while True:
                    i = 10
            nonlocal handled
            handled = True
            return True
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, handler)
        disp = scheduler.schedule_periodic(period, action, counter)
        wrapped.start()
        assert counter == 0
        assert handled is False
        assert wrapped.exc is None

    def test_catch_schedule_periodic_error_handled(self):
        if False:
            for i in range(10):
                print('nop')
        period = 0.05
        counter = 3
        handled = False

        def action(state):
            if False:
                i = 10
                return i + 15
            nonlocal counter
            if state:
                counter -= 1
                return state - 1
            if counter == 0:
                raise MyException()

        def handler(_):
            if False:
                print('Hello World!')
            nonlocal handled
            handled = True
            disp.dispose()
            return True
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, handler)
        disp = scheduler.schedule_periodic(period, action, counter)
        wrapped.start()
        assert counter == 0
        assert handled is True
        assert wrapped.exc is None

    def test_catch_schedule_periodic_error_unhandled(self):
        if False:
            i = 10
            return i + 15
        period = 0.05
        counter = 3
        handled = False

        def action(state):
            if False:
                return 10
            nonlocal counter
            if state:
                counter -= 1
                return state - 1
            if counter == 0:
                raise MyException()

        def handler(_):
            if False:
                print('Hello World!')
            nonlocal handled
            handled = True
            disp.dispose()
            return False
        wrapped = CatchSchedulerTestScheduler()
        scheduler = CatchScheduler(wrapped, handler)
        disp = scheduler.schedule_periodic(period, action, counter)
        wrapped.start()
        assert counter == 0
        assert handled is True
        assert isinstance(wrapped.exc, MyException)