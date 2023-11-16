import unittest
from reactivex import throw
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
        i = 10
        return i + 15
    raise RxException(ex)

class TestThrow(unittest.TestCase):

    def test_throw_exception_basic(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        ex = 'ex'

        def factory():
            if False:
                print('Hello World!')
            return throw(ex)
        results = scheduler.start(factory)
        assert results.messages == [on_error(200, ex)]

    def test_throw_disposed(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()

        def factory():
            if False:
                print('Hello World!')
            return throw('ex')
        results = scheduler.start(factory, disposed=200)
        assert results.messages == []

    def test_throw_observer_throws(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = throw('ex')
        xs.subscribe(lambda x: None, lambda ex: _raise('ex'), lambda : None, scheduler=scheduler)
        self.assertRaises(RxException, scheduler.start)