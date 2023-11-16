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

class RxException(Exception):
    pass

def _raise(ex):
    if False:
        while True:
            i = 10
    raise RxException(ex)

class TestGenerateWithRelativeTime(unittest.TestCase):

    def test_generate_timespan_finite(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()

        def create():
            if False:
                return 10
            return reactivex.generate_with_relative_time(0, lambda x: x <= 3, lambda x: x + 1, lambda x: x + 1)
        results = scheduler.start(create)
        assert results.messages == [on_next(201, 0), on_next(203, 1), on_next(206, 2), on_next(210, 3), on_completed(210)]

    def test_generate_timespan_throw_condition(self):
        if False:
            print('Hello World!')
        ex = 'ex'
        scheduler = TestScheduler()

        def create():
            if False:
                return 10
            return reactivex.generate_with_relative_time(0, lambda x: _raise(ex), lambda x: x + 1, lambda x: x + 1)
        results = scheduler.start(create)
        assert results.messages == [on_error(200, ex)]

    def test_generate_timespan_throw_iterate(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return reactivex.generate_with_relative_time(0, lambda x: True, lambda x: _raise(ex), lambda x: x + 1)
        results = scheduler.start(create)
        assert results.messages == [on_next(201, 0), on_error(201, ex)]

    def test_generate_timespan_throw_timemapper(self):
        if False:
            for i in range(10):
                print('nop')
        ex = 'ex'
        scheduler = TestScheduler()

        def create():
            if False:
                i = 10
                return i + 15
            return reactivex.generate_with_relative_time(0, lambda x: True, lambda x: x + 1, lambda x: _raise(ex))
        results = scheduler.start(create)
        assert results.messages == [on_error(200, ex)]

    def test_generate_timespan_dispose(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()

        def create():
            if False:
                while True:
                    i = 10
            return reactivex.generate_with_relative_time(0, lambda x: True, lambda x: x + 1, lambda x: x + 1)
        results = scheduler.start(create, disposed=210)
        assert results.messages == [on_next(201, 0), on_next(203, 1), on_next(206, 2)]

    def test_generate_datetime_offset_finite(self):
        if False:
            return 10
        scheduler = TestScheduler()

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return reactivex.generate_with_relative_time(0, lambda x: x <= 3, lambda x: x + 1, lambda x: x + 1)
        results = scheduler.start(create)
        assert results.messages == [on_next(201, 0), on_next(203, 1), on_next(206, 2), on_next(210, 3), on_completed(210)]

    def test_generate_datetime_offset_throw_condition(self):
        if False:
            while True:
                i = 10
        ex = 'ex'
        scheduler = TestScheduler()

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return reactivex.generate_with_relative_time(0, lambda x: _raise(ex), lambda x: x + 1, lambda x: x + 1)
        results = scheduler.start(create)
        assert results.messages == [on_error(200, ex)]

    def test_generate_datetime_offset_throw_iterate(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return reactivex.generate_with_relative_time(0, lambda x: True, lambda x: _raise(ex), lambda x: x + 1)
        results = scheduler.start(create)
        assert results.messages == [on_next(201, 0), on_error(201, ex)]

    def test_generate_datetime_offset_throw_time_mapper(self):
        if False:
            while True:
                i = 10
        ex = 'ex'
        scheduler = TestScheduler()

        def create():
            if False:
                return 10
            return reactivex.generate_with_relative_time(0, lambda x: True, lambda x: x + 1, lambda x: _raise(ex))
        results = scheduler.start(create)
        assert results.messages == [on_error(200, ex)]

    def test_generate_datetime_offset_dispose(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()

        def create():
            if False:
                i = 10
                return i + 15
            return reactivex.generate_with_relative_time(0, lambda x: True, lambda x: x + 1, lambda x: x + 1)
        results = scheduler.start(create, disposed=210)
        assert results.messages == [on_next(201, 0), on_next(203, 1), on_next(206, 2)]