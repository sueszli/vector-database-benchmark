import time
from datetime import timedelta
from unittest.mock import patch
import pytest
import dramatiq
from dramatiq import Message, Middleware
from dramatiq.errors import RateLimitExceeded
from dramatiq.middleware import CurrentMessage, SkipMessage
from .common import skip_on_pypy, worker

def test_actors_can_be_defined(stub_broker):
    if False:
        i = 10
        return i + 15

    @dramatiq.actor
    def add(x, y):
        if False:
            print('Hello World!')
        return x + y
    assert isinstance(add, dramatiq.Actor)

def test_actors_can_be_declared_with_actor_class(stub_broker):
    if False:
        while True:
            i = 10

    class ActorChild(dramatiq.Actor):
        pass

    @dramatiq.actor(actor_class=ActorChild)
    def add(x, y):
        if False:
            while True:
                i = 10
        return x + y
    assert isinstance(add, ActorChild)

def test_actors_can_be_assigned_predefined_options(stub_broker):
    if False:
        i = 10
        return i + 15

    @dramatiq.actor(max_retries=32)
    def add(x, y):
        if False:
            i = 10
            return i + 15
        return x + y
    assert add.options['max_retries'] == 32

def test_actors_cannot_be_assigned_arbitrary_options(stub_broker):
    if False:
        return 10
    with pytest.raises(ValueError):

        @dramatiq.actor(invalid_option=32)
        def add(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x + y

def test_actors_can_be_named(stub_broker):
    if False:
        return 10

    @dramatiq.actor(actor_name='foo')
    def add(x, y):
        if False:
            while True:
                i = 10
        return x + y
    assert add.actor_name == 'foo'

def test_actors_can_be_assigned_custom_queues(stub_broker):
    if False:
        while True:
            i = 10

    @dramatiq.actor(queue_name='foo')
    def foo():
        if False:
            for i in range(10):
                print('nop')
        pass
    assert foo.queue_name == 'foo'

def test_actors_fail_given_invalid_queue_names(stub_broker):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):

        @dramatiq.actor(queue_name='$2@!@#')
        def foo():
            if False:
                i = 10
                return i + 15
            pass

def test_actors_can_be_called(stub_broker):
    if False:
        i = 10
        return i + 15

    @dramatiq.actor
    def add(x, y):
        if False:
            for i in range(10):
                print('nop')
        return x + y
    assert add(1, 2) == 3

def test_actors_can_be_sent_messages(stub_broker):
    if False:
        for i in range(10):
            print('nop')

    @dramatiq.actor
    def add(x, y):
        if False:
            for i in range(10):
                print('nop')
        return x + y
    enqueued_message = add.send(1, 2)
    enqueued_message_data = stub_broker.queues['default'].get(timeout=1)
    assert enqueued_message == Message.decode(enqueued_message_data)

def test_actors_can_perform_work(stub_broker, stub_worker):
    if False:
        while True:
            i = 10
    database = {}

    @dramatiq.actor
    def put(key, value):
        if False:
            print('Hello World!')
        database[key] = value
    for i in range(100):
        assert put.send('key-%s' % i, i)
    stub_broker.join(put.queue_name)
    stub_worker.join()
    assert len(database) == 100

def test_actors_can_perform_work_with_kwargs(stub_broker, stub_worker):
    if False:
        while True:
            i = 10
    results = []

    @dramatiq.actor
    def add(x, y):
        if False:
            for i in range(10):
                print('nop')
        results.append(x + y)
    add.send(x=1, y=2)
    stub_broker.join(add.queue_name)
    stub_worker.join()
    assert results == [3]

@skip_on_pypy
def test_actors_can_be_assigned_time_limits(stub_broker, stub_worker):
    if False:
        return 10
    (attempts, successes) = ([], [])

    @dramatiq.actor(max_retries=0, time_limit=1000)
    def do_work():
        if False:
            for i in range(10):
                print('nop')
        attempts.append(1)
        time.sleep(3)
        successes.append(1)
    do_work.send()
    stub_broker.join(do_work.queue_name)
    stub_worker.join()
    assert sum(attempts) == 1
    assert sum(successes) == 0

@skip_on_pypy
def test_actor_messages_can_be_assigned_time_limits(stub_broker, stub_worker):
    if False:
        return 10
    (attempts, successes) = ([], [])

    @dramatiq.actor(max_retries=0)
    def do_work():
        if False:
            for i in range(10):
                print('nop')
        attempts.append(1)
        time.sleep(2)
        successes.append(1)
    do_work.send_with_options(time_limit=1000)
    stub_broker.join(do_work.queue_name)
    stub_worker.join()
    assert sum(attempts) == 1
    assert sum(successes) == 0

def test_actors_can_be_assigned_message_age_limits(stub_broker):
    if False:
        for i in range(10):
            print('nop')
    runs = []

    @dramatiq.actor(max_age=100)
    def do_work():
        if False:
            return 10
        runs.append(1)
    do_work.send()
    time.sleep(0.1)
    with worker(stub_broker, worker_timeout=100) as stub_worker:
        stub_broker.join(do_work.queue_name)
        stub_worker.join()
        assert sum(runs) == 0

def test_actor_messages_can_be_assigned_message_age_limits(stub_broker):
    if False:
        for i in range(10):
            print('nop')
    runs = []

    @dramatiq.actor()
    def do_work():
        if False:
            i = 10
            return i + 15
        runs.append(1)
    do_work.send_with_options(max_age=100)
    time.sleep(0.1)
    with worker(stub_broker, worker_timeout=100) as stub_worker:
        stub_broker.join(do_work.queue_name)
        stub_worker.join()
        assert sum(runs) == 0

def test_actors_can_delay_messages_independent_of_each_other(stub_broker, stub_worker):
    if False:
        return 10
    results = []

    @dramatiq.actor
    def append(x):
        if False:
            print('Hello World!')
        results.append(x)
    append.send_with_options(args=(1,), delay=1500)
    append.send_with_options(args=(2,), delay=timedelta(seconds=1))
    stub_broker.join(append.queue_name)
    stub_worker.join()
    assert results == [2, 1]

def test_messages_belonging_to_missing_actors_are_rejected(stub_broker, stub_worker):
    if False:
        print('Hello World!')
    message = Message(queue_name='some-queue', actor_name='some-actor', args=(), kwargs={}, options={})
    stub_broker.declare_queue('some-queue')
    stub_broker.enqueue(message)
    stub_broker.join('some-queue')
    stub_worker.join()
    assert stub_broker.dead_letters == [message]

def test_before_and_after_signal_failures_are_ignored(stub_broker, stub_worker):
    if False:
        i = 10
        return i + 15

    class BrokenMiddleware(Middleware):

        def before_process_message(self, broker, message):
            if False:
                return 10
            raise RuntimeError('before process message error')

        def after_process_message(self, broker, message, *, result=None, exception=None):
            if False:
                print('Hello World!')
            raise RuntimeError('after process message error')
    database = []

    @dramatiq.actor
    def append(x):
        if False:
            i = 10
            return i + 15
        database.append(x)
    stub_broker.add_middleware(BrokenMiddleware())
    append.send(1)
    stub_broker.join(append.queue_name)
    stub_worker.join()
    assert database == [1]

def test_middleware_can_decide_to_skip_messages(stub_broker, stub_worker):
    if False:
        return 10
    skipped_messages = []

    class SkipMiddleware(Middleware):

        def before_process_message(self, broker, message):
            if False:
                for i in range(10):
                    print('nop')
            raise SkipMessage()

        def after_skip_message(self, broker, message):
            if False:
                while True:
                    i = 10
            skipped_messages.append(1)
    stub_broker.add_middleware(SkipMiddleware())
    calls = []

    @dramatiq.actor
    def track_call():
        if False:
            print('Hello World!')
        calls.append(1)
    track_call.send()
    stub_broker.join(track_call.queue_name)
    stub_worker.join()
    assert sum(calls) == 0
    assert sum(skipped_messages) == 1

def test_workers_can_be_paused(stub_broker, stub_worker):
    if False:
        for i in range(10):
            print('nop')
    stub_worker.pause()
    calls = []

    @dramatiq.actor
    def track_call():
        if False:
            while True:
                i = 10
        calls.append(1)
    track_call.send()
    time.sleep(0.1)
    assert calls == []
    stub_worker.resume()
    stub_broker.join(track_call.queue_name)
    stub_worker.join()
    assert calls == [1]

def test_actors_can_prioritize_work(stub_broker):
    if False:
        i = 10
        return i + 15
    with worker(stub_broker, worker_timeout=100, worker_threads=1) as stub_worker:
        stub_worker.pause()
        calls = []

        @dramatiq.actor(priority=0)
        def hi():
            if False:
                print('Hello World!')
            calls.append('hi')

        @dramatiq.actor(priority=10)
        def lo():
            if False:
                return 10
            calls.append('lo')
        for _ in range(10):
            lo.send()
            hi.send()
        stub_worker.resume()
        stub_broker.join(lo.queue_name)
        stub_worker.join()
        assert calls == ['hi'] * 10 + ['lo'] * 10

def test_can_call_str_on_actors():
    if False:
        return 10

    @dramatiq.actor
    def test():
        if False:
            for i in range(10):
                print('nop')
        pass
    assert str(test) == 'Actor(test)'

def test_can_call_repr_on_actors():
    if False:
        print('Hello World!')

    @dramatiq.actor
    def test():
        if False:
            return 10
        pass
    assert repr(test) == "Actor(%(fn)r, queue_name='default', actor_name='test')" % vars(test)

def test_workers_log_rate_limit_exceeded_errors_differently(stub_broker, stub_worker):
    if False:
        return 10
    with patch('logging.Logger.debug') as debug_mock:

        @dramatiq.actor(max_retries=0)
        def raise_rate_limit_exceeded():
            if False:
                print('Hello World!')
            raise RateLimitExceeded('exceeded')
        raise_rate_limit_exceeded.send()
        stub_broker.join(raise_rate_limit_exceeded.queue_name)
        stub_worker.join()
        debug_messages = [args[0] for (_, args, _) in debug_mock.mock_calls]
        assert 'Rate limit exceeded in message %s: %s.' in debug_messages

def test_currrent_message_middleware_exposes_the_current_message(stub_broker, stub_worker):
    if False:
        while True:
            i = 10
    stub_broker.add_middleware(CurrentMessage())
    sent_messages = []
    received_messages = []

    @dramatiq.actor
    def accessor(x):
        if False:
            print('Hello World!')
        message_proxy = CurrentMessage.get_current_message()
        received_messages.append(message_proxy._message)
    sent_messages.append(accessor.send(1))
    sent_messages.append(accessor.send(2))
    stub_broker.join(accessor.queue_name)
    assert sorted(sent_messages) == sorted(received_messages)
    assert CurrentMessage.get_current_message() is None