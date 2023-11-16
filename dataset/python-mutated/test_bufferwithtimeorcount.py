import unittest
from reactivex import operators as ops
from reactivex.testing import ReactiveTest, TestScheduler
on_next = ReactiveTest.on_next
on_completed = ReactiveTest.on_completed
on_error = ReactiveTest.on_error
subscribe = ReactiveTest.subscribe
subscribed = ReactiveTest.subscribed
disposed = ReactiveTest.disposed
created = ReactiveTest.created

class TestBufferWithCount(unittest.TestCase):

    def test_buffer_with_time_or_count_basic(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(205, 1), on_next(210, 2), on_next(240, 3), on_next(280, 4), on_next(320, 5), on_next(350, 6), on_next(370, 7), on_next(420, 8), on_next(470, 9), on_completed(600))

        def create():
            if False:
                return 10
            return xs.pipe(ops.buffer_with_time_or_count(70, 3), ops.map(lambda x: ','.join([str(a) for a in x])))
        results = scheduler.start(create)
        assert results.messages == [on_next(240, '1,2,3'), on_next(310, '4'), on_next(370, '5,6,7'), on_next(440, '8'), on_next(510, '9'), on_next(580, ''), on_next(600, ''), on_completed(600)]
        assert xs.subscriptions == [subscribe(200, 600)]

    def test_buffer_with_time_or_count_error(self):
        if False:
            while True:
                i = 10
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(205, 1), on_next(210, 2), on_next(240, 3), on_next(280, 4), on_next(320, 5), on_next(350, 6), on_next(370, 7), on_next(420, 8), on_next(470, 9), on_error(600, ex))

        def create():
            if False:
                return 10
            return xs.pipe(ops.buffer_with_time_or_count(70, 3), ops.map(lambda x: ','.join([str(a) for a in x])))
        results = scheduler.start(create)
        assert results.messages == [on_next(240, '1,2,3'), on_next(310, '4'), on_next(370, '5,6,7'), on_next(440, '8'), on_next(510, '9'), on_next(580, ''), on_error(600, ex)]
        assert xs.subscriptions == [subscribe(200, 600)]

    def test_buffer_with_time_or_count_disposed(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(205, 1), on_next(210, 2), on_next(240, 3), on_next(280, 4), on_next(320, 5), on_next(350, 6), on_next(370, 7), on_next(420, 8), on_next(470, 9), on_completed(600))

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.buffer_with_time_or_count(70, 3), ops.map(lambda x: ','.join([str(a) for a in x])))
        results = scheduler.start(create, disposed=370)
        assert results.messages == [on_next(240, '1,2,3'), on_next(310, '4'), on_next(370, '5,6,7')]
        assert xs.subscriptions == [subscribe(200, 370)]