import threading
import unittest
from datetime import timedelta
from time import sleep
import pytest
from reactivex.internal.basic import default_now
from reactivex.scheduler import CurrentThreadScheduler

class TestCurrentThreadScheduler(unittest.TestCase):

    def test_currentthread_singleton(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = [CurrentThreadScheduler(), CurrentThreadScheduler.singleton(), CurrentThreadScheduler.singleton()]
        assert scheduler[0] is not scheduler[1]
        assert scheduler[1] is scheduler[2]
        gate = [threading.Semaphore(0), threading.Semaphore(0)]
        scheduler = [None, None]

        def run(idx):
            if False:
                i = 10
                return i + 15
            scheduler[idx] = CurrentThreadScheduler.singleton()
            gate[idx].release()
        for idx in (0, 1):
            threading.Thread(target=run, args=(idx,)).start()
            gate[idx].acquire()
        assert scheduler[0] is not None
        assert scheduler[1] is not None
        assert scheduler[0] is not scheduler[1]

    def test_currentthread_extend(self):
        if False:
            while True:
                i = 10

        class MyScheduler(CurrentThreadScheduler):
            pass
        scheduler = [MyScheduler(), MyScheduler.singleton(), MyScheduler.singleton(), CurrentThreadScheduler.singleton()]
        assert scheduler[0] is not scheduler[1]
        assert scheduler[1] is scheduler[2]
        assert scheduler[1] is not scheduler[3]

    def test_currentthread_now(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = CurrentThreadScheduler()
        diff = scheduler.now - default_now()
        assert abs(diff) < timedelta(milliseconds=5)

    def test_currentthread_now_units(self):
        if False:
            print('Hello World!')
        scheduler = CurrentThreadScheduler()
        diff = scheduler.now
        sleep(1.1)
        diff = scheduler.now - diff
        assert timedelta(milliseconds=1000) < diff < timedelta(milliseconds=1300)

    def test_currentthread_schedule(self):
        if False:
            while True:
                i = 10
        scheduler = CurrentThreadScheduler()
        ran = False

        def action(scheduler, state=None):
            if False:
                i = 10
                return i + 15
            nonlocal ran
            ran = True
        scheduler.schedule(action)
        assert ran is True

    def test_currentthread_schedule_block(self):
        if False:
            i = 10
            return i + 15
        scheduler = CurrentThreadScheduler()
        ran = False

        def action(scheduler, state=None):
            if False:
                return 10
            nonlocal ran
            ran = True
        t = scheduler.now
        scheduler.schedule_relative(0.2, action)
        t = scheduler.now - t
        assert ran is True
        assert t >= timedelta(seconds=0.2)

    def test_currentthread_schedule_error(self):
        if False:
            return 10
        scheduler = CurrentThreadScheduler()

        class MyException(Exception):
            pass

        def action(scheduler, state=None):
            if False:
                while True:
                    i = 10
            raise MyException()
        with pytest.raises(MyException):
            scheduler.schedule(action)

    def test_currentthread_schedule_nested(self):
        if False:
            i = 10
            return i + 15
        scheduler = CurrentThreadScheduler()
        ran = False

        def action(scheduler, state=None):
            if False:
                while True:
                    i = 10

            def inner_action(scheduler, state=None):
                if False:
                    return 10
                nonlocal ran
                ran = True
            return scheduler.schedule(inner_action)
        scheduler.schedule(action)
        assert ran is True

    def test_currentthread_schedule_nested_order(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = CurrentThreadScheduler()
        tests = []

        def outer(scheduler, state=None):
            if False:
                for i in range(10):
                    print('nop')

            def action1(scheduler, state=None):
                if False:
                    print('Hello World!')
                tests.append(1)

                def action2(scheduler, state=None):
                    if False:
                        i = 10
                        return i + 15
                    tests.append(2)
                CurrentThreadScheduler().schedule(action2)
            CurrentThreadScheduler().schedule(action1)

            def action3(scheduler, state=None):
                if False:
                    return 10
                tests.append(3)
            CurrentThreadScheduler().schedule(action3)
        scheduler.ensure_trampoline(outer)
        assert tests == [1, 2, 3]

    def test_currentthread_singleton_schedule_nested_order(self):
        if False:
            return 10
        scheduler = CurrentThreadScheduler.singleton()
        tests = []

        def outer(scheduler, state=None):
            if False:
                return 10

            def action1(scheduler, state=None):
                if False:
                    return 10
                tests.append(1)

                def action2(scheduler, state=None):
                    if False:
                        return 10
                    tests.append(2)
                scheduler.schedule(action2)
            scheduler.schedule(action1)

            def action3(scheduler, state=None):
                if False:
                    print('Hello World!')
                tests.append(3)
            scheduler.schedule(action3)
        scheduler.ensure_trampoline(outer)
        assert tests == [1, 3, 2]

    def test_currentthread_ensuretrampoline(self):
        if False:
            print('Hello World!')
        scheduler = CurrentThreadScheduler()
        (ran1, ran2) = (False, False)

        def outer_action(scheduer, state=None):
            if False:
                i = 10
                return i + 15

            def action1(scheduler, state=None):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal ran1
                ran1 = True
            scheduler.schedule(action1)

            def action2(scheduler, state=None):
                if False:
                    while True:
                        i = 10
                nonlocal ran2
                ran2 = True
            return scheduler.schedule(action2)
        scheduler.ensure_trampoline(outer_action)
        assert ran1 is True
        assert ran2 is True

    def test_currentthread_ensuretrampoline_nested(self):
        if False:
            i = 10
            return i + 15
        scheduler = CurrentThreadScheduler()
        (ran1, ran2) = (False, False)

        def outer_action(scheduler, state):
            if False:
                i = 10
                return i + 15

            def inner_action1(scheduler, state):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal ran1
                ran1 = True
            scheduler.schedule(inner_action1)

            def inner_action2(scheduler, state):
                if False:
                    print('Hello World!')
                nonlocal ran2
                ran2 = True
            return scheduler.schedule(inner_action2)
        scheduler.ensure_trampoline(outer_action)
        assert ran1 is True
        assert ran2 is True

    def test_currentthread_ensuretrampoline_and_cancel(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = CurrentThreadScheduler()
        (ran1, ran2) = (False, False)

        def outer_action(scheduler, state):
            if False:
                i = 10
                return i + 15

            def inner_action1(scheduler, state):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal ran1
                ran1 = True

                def inner_action2(scheduler, state):
                    if False:
                        print('Hello World!')
                    nonlocal ran2
                    ran2 = True
                d = scheduler.schedule(inner_action2)
                d.dispose()
            return scheduler.schedule(inner_action1)
        scheduler.ensure_trampoline(outer_action)
        assert ran1 is True
        assert ran2 is False

    def test_currentthread_ensuretrampoline_and_canceltimed(self):
        if False:
            return 10
        scheduler = CurrentThreadScheduler()
        (ran1, ran2) = (False, False)

        def outer_action(scheduler, state):
            if False:
                for i in range(10):
                    print('nop')

            def inner_action1(scheduler, state):
                if False:
                    return 10
                nonlocal ran1
                ran1 = True

                def inner_action2(scheduler, state):
                    if False:
                        i = 10
                        return i + 15
                    nonlocal ran2
                    ran2 = True
                t = scheduler.now + timedelta(seconds=0.5)
                d = scheduler.schedule_absolute(t, inner_action2)
                d.dispose()
            return scheduler.schedule(inner_action1)
        scheduler.ensure_trampoline(outer_action)
        assert ran1 is True
        assert ran2 is False