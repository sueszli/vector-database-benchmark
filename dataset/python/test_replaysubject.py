import sys

import pytest

from reactivex.internal.exceptions import DisposedException
from reactivex.subject import ReplaySubject
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


# Helper function for raising exceptions within lambdas
def _raise(ex):
    raise RxException(ex)


def test_infinite():
    scheduler = TestScheduler()

    xs = scheduler.create_hot_observable(
        on_next(70, 1),
        on_next(110, 2),
        on_next(220, 3),
        on_next(270, 4),
        on_next(340, 5),
        on_next(410, 6),
        on_next(520, 7),
        on_next(630, 8),
        on_next(710, 9),
        on_next(870, 10),
        on_next(940, 11),
        on_next(1020, 12),
    )

    subject = [None]
    subscription = [None]
    subscription1 = [None]
    subscription2 = [None]
    subscription3 = [None]

    results1 = scheduler.create_observer()
    results2 = scheduler.create_observer()
    results3 = scheduler.create_observer()

    def action1(scheduler, state=None):
        subject[0] = ReplaySubject(3, 100, scheduler)

    scheduler.schedule_absolute(100, action1)

    def action2(scheduler, state=None):
        subscription[0] = xs.subscribe(subject[0])

    scheduler.schedule_absolute(200, action2)

    def action3(scheduler, state=None):
        subscription[0].dispose()

    scheduler.schedule_absolute(1000, action3)

    def action4(scheduler, state=None):
        subscription1[0] = subject[0].subscribe(results1)

    scheduler.schedule_absolute(300, action4)

    def action5(scheduler, state=None):
        subscription2[0] = subject[0].subscribe(results2)

    scheduler.schedule_absolute(400, action5)

    def action6(scheduler, state=None):
        subscription3[0] = subject[0].subscribe(results3)

    scheduler.schedule_absolute(900, action6)

    def action7(scheduler, state=None):
        subscription1[0].dispose()

    scheduler.schedule_absolute(600, action7)

    def action8(scheduler, state=None):
        subscription2[0].dispose()

    scheduler.schedule_absolute(700, action8)

    def action9(scheduler, state=None):
        subscription1[0].dispose()

    scheduler.schedule_absolute(800, action9)

    def action10(scheduler, state=None):
        subscription3[0].dispose()

    scheduler.schedule_absolute(950, action10)

    scheduler.start()

    assert results1.messages == [
        on_next(300, 3),
        on_next(300, 4),
        on_next(340, 5),
        on_next(410, 6),
        on_next(520, 7),
    ]

    assert results2.messages == [
        on_next(400, 5),
        on_next(410, 6),
        on_next(520, 7),
        on_next(630, 8),
    ]

    assert results3.messages == [on_next(900, 10), on_next(940, 11)]


def test_infinite2():
    scheduler = TestScheduler()

    xs = scheduler.create_hot_observable(
        on_next(70, 1),
        on_next(110, 2),
        on_next(220, 3),
        on_next(270, 4),
        on_next(280, -1),
        on_next(290, -2),
        on_next(340, 5),
        on_next(410, 6),
        on_next(520, 7),
        on_next(630, 8),
        on_next(710, 9),
        on_next(870, 10),
        on_next(940, 11),
        on_next(1020, 12),
    )

    subject = [None]
    subscription = [None]
    subscription1 = [None]
    subscription2 = [None]
    subscription3 = [None]

    results1 = scheduler.create_observer()
    results2 = scheduler.create_observer()
    results3 = scheduler.create_observer()

    def action1(scheduler, state=None):
        subject[0] = ReplaySubject(3, 100, scheduler)

    scheduler.schedule_absolute(100, action1)

    def action2(scheduler, state=None):
        subscription[0] = xs.subscribe(subject[0])

    scheduler.schedule_absolute(200, action2)

    def action3(scheduler, state=None):
        subscription[0].dispose()

    scheduler.schedule_absolute(1000, action3)

    def action4(scheduler, state=None):
        subscription1[0] = subject[0].subscribe(results1)

    scheduler.schedule_absolute(300, action4)

    def action5(scheduler, state=None):
        subscription2[0] = subject[0].subscribe(results2)

    scheduler.schedule_absolute(400, action5)

    def action6(scheduler, state=None):
        subscription3[0] = subject[0].subscribe(results3)

    scheduler.schedule_absolute(900, action6)

    def action7(scheduler, state=None):
        subscription1[0].dispose()

    scheduler.schedule_absolute(600, action7)

    def action8(scheduler, state=None):
        subscription2[0].dispose()

    scheduler.schedule_absolute(700, action8)

    def action9(scheduler, state=None):
        subscription1[0].dispose()

    scheduler.schedule_absolute(800, action9)

    def action10(scheduler, state=None):
        subscription3[0].dispose()

    scheduler.schedule_absolute(950, action10)

    scheduler.start()

    assert results1.messages == [
        on_next(300, 4),
        on_next(300, -1),
        on_next(300, -2),
        on_next(340, 5),
        on_next(410, 6),
        on_next(520, 7),
    ]

    assert results2.messages == [
        on_next(400, 5),
        on_next(410, 6),
        on_next(520, 7),
        on_next(630, 8),
    ]

    assert results3.messages == [on_next(900, 10), on_next(940, 11)]


def test_finite():
    scheduler = TestScheduler()

    xs = scheduler.create_hot_observable(
        on_next(70, 1),
        on_next(110, 2),
        on_next(220, 3),
        on_next(270, 4),
        on_next(340, 5),
        on_next(410, 6),
        on_next(520, 7),
        on_completed(630),
        on_next(640, 9),
        on_completed(650),
        on_error(660, "ex"),
    )

    subject = [None]
    subscription = [None]
    subscription1 = [None]
    subscription2 = [None]
    subscription3 = [None]

    results1 = scheduler.create_observer()
    results2 = scheduler.create_observer()
    results3 = scheduler.create_observer()

    def action1(scheduler, state=None):
        subject[0] = ReplaySubject(3, 100, scheduler)

    scheduler.schedule_absolute(100, action1)

    def action3(scheduler, state=None):
        subscription[0] = xs.subscribe(subject[0])

    scheduler.schedule_absolute(200, action3)

    def action4(scheduler, state=None):
        subscription[0].dispose()

    scheduler.schedule_absolute(1000, action4)

    def action5(scheduler, state=None):
        subscription1[0] = subject[0].subscribe(results1)

    scheduler.schedule_absolute(300, action5)

    def action6(scheduler, state=None):
        subscription2[0] = subject[0].subscribe(results2)

    scheduler.schedule_absolute(400, action6)

    def action7(scheduler, state=None):
        subscription3[0] = subject[0].subscribe(results3)

    scheduler.schedule_absolute(900, action7)

    def action8(scheduler, state=None):
        subscription1[0].dispose()

    scheduler.schedule_absolute(600, action8)

    def action9(scheduler, state=None):
        subscription2[0].dispose()

    scheduler.schedule_absolute(700, action9)

    def action10(scheduler, state=None):
        subscription1[0].dispose()

    scheduler.schedule_absolute(800, action10)

    def action11(scheduler, state=None):
        subscription3[0].dispose()

    scheduler.schedule_absolute(950, action11)

    scheduler.start()

    assert results1.messages == [
        on_next(300, 3),
        on_next(300, 4),
        on_next(340, 5),
        on_next(410, 6),
        on_next(520, 7),
    ]

    assert results2.messages == [
        on_next(400, 5),
        on_next(410, 6),
        on_next(520, 7),
        on_completed(630),
    ]

    assert results3.messages == [on_completed(900)]


def test_error():
    scheduler = TestScheduler()

    ex = RxException("ex")

    xs = scheduler.create_hot_observable(
        on_next(70, 1),
        on_next(110, 2),
        on_next(220, 3),
        on_next(270, 4),
        on_next(340, 5),
        on_next(410, 6),
        on_next(520, 7),
        on_error(630, ex),
        on_next(640, 9),
        on_completed(650),
        on_error(660, RxException("ex")),
    )

    subject = [None]
    subscription = [None]
    subscription1 = [None]
    subscription2 = [None]
    subscription3 = [None]

    results1 = scheduler.create_observer()
    results2 = scheduler.create_observer()
    results3 = scheduler.create_observer()

    def action1(scheduler, state=None):
        subject[0] = ReplaySubject(3, 100, scheduler)

    scheduler.schedule_absolute(100, action1)

    def action2(scheduler, state=None):
        subscription[0] = xs.subscribe(subject[0])

    scheduler.schedule_absolute(200, action2)

    def action3(scheduler, state=None):
        subscription[0].dispose()

    scheduler.schedule_absolute(1000, action3)

    def action4(scheduler, state=None):
        subscription1[0] = subject[0].subscribe(results1)

    scheduler.schedule_absolute(300, action4)

    def action5(scheduler, state=None):
        subscription2[0] = subject[0].subscribe(results2)

    scheduler.schedule_absolute(400, action5)

    def action6(scheduler, state=None):
        subscription3[0] = subject[0].subscribe(results3)

    scheduler.schedule_absolute(900, action6)

    def action7(scheduler, state=None):
        subscription1[0].dispose()

    scheduler.schedule_absolute(600, action7)

    def action8(scheduler, state=None):
        subscription2[0].dispose()

    scheduler.schedule_absolute(700, action8)

    def action9(scheduler, state=None):
        subscription1[0].dispose()

    scheduler.schedule_absolute(800, action9)

    def action10(scheduler, state=None):
        subscription3[0].dispose()

    scheduler.schedule_absolute(950, action10)

    scheduler.start()

    assert results1.messages == [
        on_next(300, 3),
        on_next(300, 4),
        on_next(340, 5),
        on_next(410, 6),
        on_next(520, 7),
    ]

    assert results2.messages == [
        on_next(400, 5),
        on_next(410, 6),
        on_next(520, 7),
        on_error(630, ex),
    ]

    assert results3.messages == [on_error(900, ex)]


def test_canceled():
    scheduler = TestScheduler()

    xs = scheduler.create_hot_observable(
        on_completed(630),
        on_next(640, 9),
        on_completed(650),
        on_error(660, RxException()),
    )

    subject = [None]
    subscription = [None]
    subscription1 = [None]
    subscription2 = [None]
    subscription3 = [None]

    results1 = scheduler.create_observer()
    results2 = scheduler.create_observer()
    results3 = scheduler.create_observer()

    def action1(scheduler, state=None):
        subject[0] = ReplaySubject(3, 100, scheduler)

    scheduler.schedule_absolute(100, action1)

    def action2(scheduler, state=None):
        subscription[0] = xs.subscribe(subject[0])

    scheduler.schedule_absolute(200, action2)

    def action3(scheduler, state=None):
        subscription[0].dispose()

    scheduler.schedule_absolute(1000, action3)

    def action4(scheduler, state=None):
        subscription1[0] = subject[0].subscribe(results1)

    scheduler.schedule_absolute(300, action4)

    def action5(scheduler, state=None):
        subscription2[0] = subject[0].subscribe(results2)

    scheduler.schedule_absolute(400, action5)

    def action6(scheduler, state=None):
        subscription3[0] = subject[0].subscribe(results3)

    scheduler.schedule_absolute(900, action6)

    def action7(scheduler, state=None):
        subscription1[0].dispose()

    scheduler.schedule_absolute(600, action7)

    def action8(scheduler, state=None):
        subscription2[0].dispose()

    scheduler.schedule_absolute(700, action8)

    def action9(scheduler, state=None):
        subscription1[0].dispose()

    scheduler.schedule_absolute(800, action9)

    def action10(scheduler, state=None):
        subscription3[0].dispose()

    scheduler.schedule_absolute(950, action10)

    scheduler.start()

    assert results1.messages == []

    assert results2.messages == [on_completed(630)]

    assert results3.messages == [on_completed(900)]


def test_subject_disposed():
    scheduler = TestScheduler()

    subject = [None]
    subscription1 = [None]
    subscription2 = [None]
    subscription3 = [None]

    results1 = scheduler.create_observer()
    results2 = scheduler.create_observer()
    results3 = scheduler.create_observer()

    def action1(scheduler, state=None):
        subject[0] = ReplaySubject(scheduler=scheduler)

    scheduler.schedule_absolute(100, action1)

    def action2(scheduler, state=None):
        subscription1[0] = subject[0].subscribe(results1)

    scheduler.schedule_absolute(200, action2)

    def action3(scheduler, state=None):
        subscription2[0] = subject[0].subscribe(results2)

    scheduler.schedule_absolute(300, action3)

    def action4(scheduler, state=None):
        subscription3[0] = subject[0].subscribe(results3)

    scheduler.schedule_absolute(400, action4)

    def action5(scheduler, state=None):
        subscription1[0].dispose()

    scheduler.schedule_absolute(500, action5)

    def action6(scheduler, state=None):
        subject[0].dispose()

    scheduler.schedule_absolute(600, action6)

    def action7(scheduler, state=None):
        subscription2[0].dispose()

    scheduler.schedule_absolute(700, action7)

    def action8(scheduler, state=None):
        subscription3[0].dispose()

    scheduler.schedule_absolute(800, action8)

    def action9(scheduler, state=None):
        subject[0].on_next(1)

    scheduler.schedule_absolute(150, action9)

    def action10(scheduler, state=None):
        subject[0].on_next(2)

    scheduler.schedule_absolute(250, action10)

    def action11(scheduler, state=None):
        subject[0].on_next(3)

    scheduler.schedule_absolute(350, action11)

    def action12(scheduler, state=None):
        subject[0].on_next(4)

    scheduler.schedule_absolute(450, action12)

    def action13(scheduler, state=None):
        subject[0].on_next(5)

    scheduler.schedule_absolute(550, action13)

    def action14(scheduler, state=None):
        with pytest.raises(DisposedException):
            subject[0].on_next(6)

    scheduler.schedule_absolute(650, action14)

    def action15(scheduler, state=None):
        with pytest.raises(DisposedException):
            subject[0].on_completed()

    scheduler.schedule_absolute(750, action15)

    def action16(scheduler, state=None):
        with pytest.raises(DisposedException):
            subject[0].on_error(Exception())

    scheduler.schedule_absolute(850, action16)

    def action17(scheduler, state=None):
        with pytest.raises(DisposedException):
            subject[0].subscribe(None)

    scheduler.schedule_absolute(950, action17)

    scheduler.start()

    assert results1.messages == [
        on_next(200, 1),
        on_next(250, 2),
        on_next(350, 3),
        on_next(450, 4),
    ]

    assert results2.messages == [
        on_next(300, 1),
        on_next(300, 2),
        on_next(350, 3),
        on_next(450, 4),
        on_next(550, 5),
    ]

    assert results3.messages == [
        on_next(400, 1),
        on_next(400, 2),
        on_next(400, 3),
        on_next(450, 4),
        on_next(550, 5),
    ]


def test_replay_subject_dies_out():
    scheduler = TestScheduler()

    xs = scheduler.create_hot_observable(
        on_next(70, 1),
        on_next(110, 2),
        on_next(220, 3),
        on_next(270, 4),
        on_next(340, 5),
        on_next(410, 6),
        on_next(520, 7),
        on_completed(580),
    )

    subject = [None]

    results1 = scheduler.create_observer()
    results2 = scheduler.create_observer()
    results3 = scheduler.create_observer()
    results4 = scheduler.create_observer()

    def action1(scheduler, state=None):
        subject[0] = ReplaySubject(sys.maxsize, 100, scheduler)

    scheduler.schedule_absolute(100, action1)

    def action2(scheduler, state=None):
        xs.subscribe(subject[0])

    scheduler.schedule_absolute(200, action2)

    def action3(scheduler, state=None):
        subject[0].subscribe(results1)

    scheduler.schedule_absolute(300, action3)

    def action4(scheduler, state=None):
        subject[0].subscribe(results2)

    scheduler.schedule_absolute(400, action4)

    def action5(scheduler, state=None):
        subject[0].subscribe(results3)

    scheduler.schedule_absolute(600, action5)

    def action6(scheduler, state=None):
        subject[0].subscribe(results4)

    scheduler.schedule_absolute(900, action6)

    scheduler.start()

    assert results1.messages == [
        on_next(300, 3),
        on_next(300, 4),
        on_next(340, 5),
        on_next(410, 6),
        on_next(520, 7),
        on_completed(580),
    ]

    assert results2.messages == [
        on_next(400, 5),
        on_next(410, 6),
        on_next(520, 7),
        on_completed(580),
    ]

    assert results3.messages == [on_next(600, 7), on_completed(600)]

    assert results4.messages == [on_completed(900)]
