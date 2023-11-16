import unittest
import reactivex
from reactivex.testing import ReactiveTest, TestScheduler
on_next = ReactiveTest.on_next
on_completed = ReactiveTest.on_completed
on_error = ReactiveTest.on_error
subscribe = ReactiveTest.subscribe
subscribed = ReactiveTest.subscribed
disposed = ReactiveTest.disposed
created = ReactiveTest.created

class TestToAsync(unittest.TestCase):

    def test_to_async_context(self):
        if False:
            while True:
                i = 10

        class Context:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.value = 42

            def func(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.value + x
        scheduler = TestScheduler()

        def create():
            if False:
                while True:
                    i = 10
            context = Context()
            return reactivex.to_async(context.func, scheduler)(42)
        res = scheduler.start(create)
        assert res.messages == [on_next(200, 84), on_completed(200)]

    def test_to_async0(self):
        if False:
            return 10
        scheduler = TestScheduler()

        def create():
            if False:
                i = 10
                return i + 15

            def func():
                if False:
                    for i in range(10):
                        print('nop')
                return 0
            return reactivex.to_async(func, scheduler)()
        res = scheduler.start(create)
        assert res.messages == [on_next(200, 0), on_completed(200)]

    def test_to_async1(self):
        if False:
            return 10
        scheduler = TestScheduler()

        def create():
            if False:
                print('Hello World!')

            def func(x):
                if False:
                    return 10
                return x
            return reactivex.to_async(func, scheduler)(1)
        res = scheduler.start(create)
        assert res.messages == [on_next(200, 1), on_completed(200)]

    def test_to_async2(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()

        def create():
            if False:
                return 10

            def func(x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return x + y
            return reactivex.to_async(func, scheduler)(1, 2)
        res = scheduler.start(create)
        assert res.messages == [on_next(200, 3), on_completed(200)]

    def test_to_async3(self):
        if False:
            return 10
        scheduler = TestScheduler()

        def create():
            if False:
                print('Hello World!')

            def func(x, y, z):
                if False:
                    i = 10
                    return i + 15
                return x + y + z
            return reactivex.to_async(func, scheduler)(1, 2, 3)
        res = scheduler.start(create)
        assert res.messages == [on_next(200, 6), on_completed(200)]

    def test_to_async4(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()

        def create():
            if False:
                while True:
                    i = 10

            def func(a, b, c, d):
                if False:
                    print('Hello World!')
                return a + b + c + d
            return reactivex.to_async(func, scheduler)(1, 2, 3, 4)
        res = scheduler.start(create)
        assert res.messages == [on_next(200, 10), on_completed(200)]

    def test_to_async_error0(self):
        if False:
            print('Hello World!')
        ex = Exception()
        scheduler = TestScheduler()

        def create():
            if False:
                print('Hello World!')

            def func():
                if False:
                    while True:
                        i = 10
                raise ex
            return reactivex.to_async(func, scheduler)()
        res = scheduler.start(create)
        assert res.messages == [on_error(200, ex)]

    def test_to_async_error1(self):
        if False:
            print('Hello World!')
        ex = Exception()
        scheduler = TestScheduler()

        def create():
            if False:
                for i in range(10):
                    print('nop')

            def func(a):
                if False:
                    for i in range(10):
                        print('nop')
                raise ex
            return reactivex.to_async(func, scheduler)(1)
        res = scheduler.start(create)
        assert res.messages == [on_error(200, ex)]

    def test_to_async_error2(self):
        if False:
            for i in range(10):
                print('nop')
        ex = Exception()
        scheduler = TestScheduler()

        def create():
            if False:
                i = 10
                return i + 15

            def func(a, b):
                if False:
                    for i in range(10):
                        print('nop')
                raise ex
            return reactivex.to_async(func, scheduler)(1, 2)
        res = scheduler.start(create)
        assert res.messages == [on_error(200, ex)]

    def test_to_async_error3(self):
        if False:
            return 10
        ex = Exception()
        scheduler = TestScheduler()

        def create():
            if False:
                while True:
                    i = 10

            def func(a, b, c):
                if False:
                    print('Hello World!')
                raise ex
            return reactivex.to_async(func, scheduler)(1, 2, 3)
        res = scheduler.start(create)
        assert res.messages == [on_error(200, ex)]

    def test_to_async_error4(self):
        if False:
            print('Hello World!')
        ex = Exception()
        scheduler = TestScheduler()

        def create():
            if False:
                while True:
                    i = 10

            def func(a, b, c, d):
                if False:
                    return 10
                raise ex
            return reactivex.to_async(func, scheduler)(1, 2, 3, 4)
        res = scheduler.start(create)
        assert res.messages == [on_error(200, ex)]