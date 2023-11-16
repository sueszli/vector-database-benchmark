from typing import Any
from reactivex.abc import ObserverBase
from reactivex.notification import OnCompleted, OnError, OnNext
from reactivex.testing import ReactiveTest, TestScheduler
on_next = ReactiveTest.on_next
on_completed = ReactiveTest.on_completed
on_error = ReactiveTest.on_error
subscribe = ReactiveTest.subscribe
subscribed = ReactiveTest.subscribed
disposed = ReactiveTest.disposed
created = ReactiveTest.created

def test_on_next_ctor_and_props():
    if False:
        while True:
            i = 10
    n = OnNext(42)
    assert 'N' == n.kind
    assert n.has_value
    assert 42 == n.value
    assert not hasattr(n, 'exception')

def test_on_next_equality():
    if False:
        return 10
    n1 = OnNext(42)
    n2 = OnNext(42)
    n3 = OnNext(24)
    n4 = OnCompleted()
    assert n1.equals(n1)
    assert n1.equals(n2)
    assert n2.equals(n1)
    assert not n1.equals(None)
    assert not n1.equals(n3)
    assert not n3.equals(n1)
    assert not n1.equals(n4)
    assert not n4.equals(n1)

def test_on_next_tostring():
    if False:
        i = 10
        return i + 15
    n1 = OnNext(42)
    assert 'OnNext' in str(n1)
    assert '42' in str(n1)

class CheckOnNextObserver(ObserverBase):

    def __init__(self):
        if False:
            return 10
        super(CheckOnNextObserver, self).__init__()
        self.value = None

    def on_next(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.value = value
        return self.value

    def on_error(self, error):
        if False:
            return 10
        raise NotImplementedError

    def on_completed(self):
        if False:
            while True:
                i = 10

        def func():
            if False:
                for i in range(10):
                    print('nop')
            raise NotImplementedError
        return func

def test_on_next_accept_observer():
    if False:
        return 10
    con = CheckOnNextObserver()
    n1 = OnNext(42)
    n1.accept(con)
    assert con.value == 42

class AcceptObserver(ObserverBase):

    def __init__(self, on_next, on_error, on_completed):
        if False:
            i = 10
            return i + 15
        self._on_next = on_next
        self._on_error = on_error
        self._on_completed = on_completed

    def on_next(self, value):
        if False:
            for i in range(10):
                print('nop')
        return self._on_next(value)

    def on_error(self, exception):
        if False:
            return 10
        return self._on_error(exception)

    def on_completed(self):
        if False:
            return 10
        return self._on_completed()

def test_on_next_accept_observer_with_result():
    if False:
        for i in range(10):
            print('nop')
    n1 = OnNext(42)

    def on_next(x):
        if False:
            i = 10
            return i + 15
        return 'OK'

    def on_error(err):
        if False:
            while True:
                i = 10
        assert False

    def on_completed():
        if False:
            return 10
        assert False
    res = n1.accept(AcceptObserver(on_next, on_error, on_completed))
    assert 'OK' == res

def test_on_next_accept_action():
    if False:
        while True:
            i = 10
    obs = [False]
    n1 = OnNext(42)

    def on_next(x):
        if False:
            return 10
        obs[0] = True
        return obs[0]

    def on_error(err):
        if False:
            print('Hello World!')
        assert False

    def on_completed():
        if False:
            for i in range(10):
                print('nop')
        assert False
    n1.accept(on_next, on_error, on_completed)
    assert obs[0]

def test_on_next_accept_action_with_result():
    if False:
        while True:
            i = 10
    n1 = OnNext(42)

    def on_next(x):
        if False:
            print('Hello World!')
        return 'OK'

    def on_error(err):
        if False:
            return 10
        assert False

    def on_completed():
        if False:
            i = 10
            return i + 15
        assert False
    res = n1.accept(on_next, on_error, on_completed)
    assert 'OK' == res

def test_throw_ctor_and_props():
    if False:
        return 10
    e = 'e'
    n = OnError(e)
    assert 'E' == n.kind
    assert not n.has_value
    assert e == str(n.exception)

def test_throw_equality():
    if False:
        i = 10
        return i + 15
    ex1 = 'ex1'
    ex2 = 'ex2'
    n1 = OnError(ex1)
    n2 = OnError(ex1)
    n3 = OnError(ex2)
    n4 = OnCompleted()
    assert n1.equals(n1)
    assert n1.equals(n2)
    assert n2.equals(n1)
    assert not n1.equals(None)
    assert not n1.equals(n3)
    assert not n3.equals(n1)
    assert not n1.equals(n4)
    assert not n4.equals(n1)

def test_throw_tostring():
    if False:
        print('Hello World!')
    ex = 'ex'
    n1 = OnError(ex)
    assert 'OnError' in str(n1)
    assert 'ex' in str(n1)

class CheckOnErrorObserver(ObserverBase[Any]):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(CheckOnErrorObserver, self).__init__()
        self.error = None

    def on_next(self, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def on_error(self, error: Exception) -> None:
        if False:
            return 10
        self.error = str(error)

    def on_completed(self) -> None:
        if False:
            return 10
        raise NotImplementedError()

def test_throw_accept_observer():
    if False:
        return 10
    ex = 'ex'
    obs = CheckOnErrorObserver()
    n1 = OnError(ex)
    n1.accept(obs)
    assert ex == obs.error

def test_throw_accept_observer_with_result():
    if False:
        while True:
            i = 10
    ex = 'ex'
    n1 = OnError(ex)

    def on_next(x):
        if False:
            for i in range(10):
                print('nop')
        assert False
        return None

    def on_error(ex):
        if False:
            while True:
                i = 10
        return 'OK'

    def on_completed():
        if False:
            while True:
                i = 10
        assert False
        return None
    res = n1.accept(AcceptObserver(on_next, on_error, on_completed))
    assert 'OK' == res

def test_throw_accept_action():
    if False:
        i = 10
        return i + 15
    ex = 'ex'
    obs = [False]
    n1 = OnError(ex)

    def on_next(x):
        if False:
            while True:
                i = 10
        assert False
        return None

    def on_error(ex):
        if False:
            return 10
        obs[0] = True
        return obs[0]

    def on_completed():
        if False:
            return 10
        assert False
        return None
    n1.accept(on_next, on_error, on_completed)
    assert obs[0]

def test_throw_accept_action_with_result():
    if False:
        for i in range(10):
            print('nop')
    ex = 'ex'
    n1 = OnError(ex)

    def on_next(x):
        if False:
            return 10
        assert False
        return None

    def on_error(ex):
        if False:
            print('Hello World!')
        return 'OK'

    def on_completed():
        if False:
            return 10
        assert False
        return None
    res = n1.accept(on_next, on_error, on_completed)
    assert 'OK' == res

def test_close_ctor_and_props():
    if False:
        print('Hello World!')
    n = OnCompleted()
    assert 'C' == n.kind
    assert not n.has_value
    assert not hasattr(n, 'exception')

def test_close_equality():
    if False:
        for i in range(10):
            print('nop')
    n1 = OnCompleted()
    n2 = OnCompleted()
    n3 = OnNext(2)
    assert n1.equals(n1)
    assert n1.equals(n2)
    assert n2.equals(n1)
    assert not n1.equals(None)
    assert not n1.equals(n3)
    assert not n3.equals(n1)

def test_close_tostring():
    if False:
        for i in range(10):
            print('nop')
    n1 = OnCompleted()
    assert 'OnCompleted' in str(n1)

class CheckOnCompletedObserver(ObserverBase):

    def __init__(self):
        if False:
            print('Hello World!')
        super(CheckOnCompletedObserver, self).__init__()
        self.completed = False

    def on_next(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def on_error(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def on_completed(self):
        if False:
            i = 10
            return i + 15
        self.completed = True

def test_close_accept_observer():
    if False:
        print('Hello World!')
    obs = CheckOnCompletedObserver()
    n1 = OnCompleted()
    n1.accept(obs)
    assert obs.completed

def test_close_accept_observer_with_result():
    if False:
        return 10
    n1 = OnCompleted()

    def on_next(x):
        if False:
            i = 10
            return i + 15
        assert False
        return None

    def on_error(err):
        if False:
            return 10
        assert False
        return None

    def on_completed():
        if False:
            i = 10
            return i + 15
        return 'OK'
    res = n1.accept(AcceptObserver(on_next, on_error, on_completed))
    assert 'OK' == res

def test_close_accept_action():
    if False:
        while True:
            i = 10
    obs = [False]
    n1 = OnCompleted()

    def on_next(x):
        if False:
            i = 10
            return i + 15
        assert False
        return None

    def on_error(ex):
        if False:
            print('Hello World!')
        assert False
        return None

    def on_completed():
        if False:
            for i in range(10):
                print('nop')
        obs[0] = True
        return obs[0]
    n1.accept(on_next, on_error, on_completed)
    assert obs[0]

def test_close_accept_action_with_result():
    if False:
        i = 10
        return i + 15
    n1 = OnCompleted()

    def on_next(x):
        if False:
            return 10
        assert False
        return None

    def on_error(ex):
        if False:
            while True:
                i = 10
        assert False
        return None

    def on_completed():
        if False:
            for i in range(10):
                print('nop')
        return 'OK'
    res = n1.accept(on_next, on_error, on_completed)
    assert 'OK' == res

def test_to_observable_empty():
    if False:
        return 10
    scheduler = TestScheduler()

    def create():
        if False:
            print('Hello World!')
        return OnCompleted().to_observable(scheduler)
    res = scheduler.start(create)
    assert res.messages == [ReactiveTest.on_completed(200)]

def test_to_observable_return():
    if False:
        i = 10
        return i + 15
    scheduler = TestScheduler()

    def create():
        if False:
            return 10
        return OnNext(42).to_observable(scheduler)
    res = scheduler.start(create)
    assert res.messages == [ReactiveTest.on_next(200, 42), ReactiveTest.on_completed(200)]

def test_to_observable_on_error():
    if False:
        while True:
            i = 10
    ex = 'ex'
    scheduler = TestScheduler()

    def create():
        if False:
            while True:
                i = 10
        return OnError(ex).to_observable(scheduler)
    res = scheduler.start(create)
    assert res.messages == [ReactiveTest.on_error(200, ex)]