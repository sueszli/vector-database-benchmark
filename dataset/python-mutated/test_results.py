import time
from unittest.mock import patch
import pytest
import dramatiq
from dramatiq.message import Message
from dramatiq.middleware import Middleware, SkipMessage
from dramatiq.results import ResultFailure, ResultMissing, Results, ResultTimeout
from dramatiq.results.backends import StubBackend

def test_actors_can_store_results(stub_broker, stub_worker, result_backend):
    if False:
        for i in range(10):
            print('nop')
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def do_work():
        if False:
            print('Hello World!')
        return 42
    message = do_work.send()
    result = result_backend.get_result(message, block=True)
    assert result == 42

def test_actors_results_are_backwards_compatible(stub_broker, stub_worker, result_backend):
    if False:
        return 10
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def do_work():
        if False:
            print('Hello World!')
        return 42
    message = do_work.message()
    message_key = result_backend.build_message_key(message)
    result_backend._store(message_key, 42, 3600000)
    result = result_backend.get_result(message, block=True)
    assert result == 42

def test_actors_can_store_exceptions(stub_broker, stub_worker, result_backend):
    if False:
        for i in range(10):
            print('nop')
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True, max_retries=0)
    def do_work():
        if False:
            print('Hello World!')
        raise RuntimeError('failed')
    message = do_work.send()
    with pytest.raises(ResultFailure) as e:
        result_backend.get_result(message, block=True)
    assert str(e.value) == 'actor raised RuntimeError: failed'
    assert e.value.orig_exc_type == 'RuntimeError'
    assert e.value.orig_exc_msg == 'failed'

def test_retrieving_a_result_can_raise_result_missing(stub_broker, stub_worker, result_backend):
    if False:
        print('Hello World!')
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def do_work():
        if False:
            for i in range(10):
                print('nop')
        time.sleep(0.2)
        return 42
    message = do_work.send()
    with pytest.raises(ResultMissing):
        result_backend.get_result(message)

def test_retrieving_a_result_can_time_out(stub_broker, stub_worker, result_backend):
    if False:
        while True:
            i = 10
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def do_work():
        if False:
            print('Hello World!')
        time.sleep(0.2)
        return 42
    message = do_work.send()
    with pytest.raises(ResultTimeout):
        result_backend.get_result(message, block=True, timeout=100)

def test_messages_can_get_results_from_backend(stub_broker, stub_worker, result_backend):
    if False:
        for i in range(10):
            print('nop')
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def do_work():
        if False:
            while True:
                i = 10
        return 42
    message = do_work.send()
    assert message.get_result(backend=result_backend, block=True) == 42

def test_messages_can_get_results_from_inferred_backend(stub_broker, stub_worker, result_backend):
    if False:
        i = 10
        return i + 15
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def do_work():
        if False:
            for i in range(10):
                print('nop')
        return 42
    message = do_work.send()
    assert message.get_result(block=True) == 42

def test_messages_without_actor_not_crashing_lookup_options(stub_broker, redis_result_backend):
    if False:
        print('Hello World!')
    message = Message(queue_name='default', actor_name='idontexist', args=(), kwargs={}, options={})
    assert Results(backend=redis_result_backend).after_nack(stub_broker, message) is None

def test_messages_can_fail_to_get_results_if_there_is_no_backend(stub_broker, stub_worker):
    if False:
        while True:
            i = 10

    @dramatiq.actor
    def do_work():
        if False:
            for i in range(10):
                print('nop')
        return 42
    message = do_work.send()
    with pytest.raises(RuntimeError):
        message.get_result()

def test_actor_no_warning_when_returns_none(stub_broker, stub_worker):
    if False:
        for i in range(10):
            print('nop')
    with patch('logging.Logger.warning') as warning_mock:

        @dramatiq.actor
        def nothing():
            if False:
                while True:
                    i = 10
            pass
        nothing.send()
        stub_broker.join(nothing.queue_name)
        stub_worker.join()
        warning_messages = [args[0] for (_, args, _) in warning_mock.mock_calls]
        assert not any(('Consider adding the Results middleware' in x for x in warning_messages))

def test_actor_warning_when_returns_result_and_no_results_middleware_present(stub_broker, stub_worker):
    if False:
        i = 10
        return i + 15
    with patch('logging.Logger.warning') as warning_mock:

        @dramatiq.actor
        def always_1():
            if False:
                print('Hello World!')
            return 1
        always_1.send()
        stub_broker.join(always_1.queue_name)
        stub_worker.join()
        warning_messages = [args[0] for (_, args, _) in warning_mock.mock_calls]
        assert any(('Consider adding the Results middleware' in x for x in warning_messages))

def test_actor_warning_when_returns_result_and_store_results_is_not_set(stub_broker, stub_worker):
    if False:
        return 10
    stub_broker.add_middleware(Results(backend=StubBackend()))
    with patch('logging.Logger.warning') as warning_mock:

        @dramatiq.actor
        def always_1():
            if False:
                i = 10
                return i + 15
            return 1
        always_1.send()
        stub_broker.join(always_1.queue_name)
        stub_worker.join()
        warning_messages = [args[0] for (_, args, _) in warning_mock.mock_calls]
        assert any(('the value has been discarded' in x for x in warning_messages))

def test_actor_no_warning_when_returns_result_while_piping_and_store_results_is_not_set(stub_broker, stub_worker):
    if False:
        while True:
            i = 10
    stub_broker.add_middleware(Results(backend=StubBackend()))
    with patch('logging.Logger.warning') as warning_mock:

        @dramatiq.actor
        def always_1():
            if False:
                return 10
            return 1

        @dramatiq.actor
        def noop(x):
            if False:
                return 10
            pass
        (always_1.message() | noop.message()).run()
        stub_broker.join(always_1.queue_name)
        stub_worker.join()
        warning_messages = [args[0] for (_, args, _) in warning_mock.mock_calls]
        assert not any(('the value has been discarded' in x for x in warning_messages))

def test_actor_no_warning_when_returns_result_while_piping(stub_broker, stub_worker):
    if False:
        for i in range(10):
            print('nop')
    with patch('logging.Logger.warning') as warning_mock:

        @dramatiq.actor
        def always_1():
            if False:
                i = 10
                return i + 15
            return 1

        @dramatiq.actor
        def noop(x):
            if False:
                while True:
                    i = 10
            pass
        (always_1.message() | noop.message()).run()
        stub_broker.join(noop.queue_name)
        stub_worker.join()
        warning_messages = [args[0] for (_, args, _) in warning_mock.mock_calls]
        assert not any(('Consider adding the Results middleware' in x for x in warning_messages))

def test_actor_no_warning_when_returns_result_and_results_middleware_present(stub_broker, stub_worker, result_backend):
    if False:
        i = 10
        return i + 15
    stub_broker.add_middleware(Results(backend=result_backend))
    with patch('logging.Logger.warning') as warning_mock:

        @dramatiq.actor(store_results=True)
        def always_1():
            if False:
                for i in range(10):
                    print('nop')
            return 1
        always_1.send()
        stub_broker.join(always_1.queue_name)
        stub_worker.join()
        warning_messages = [args[0] for (_, args, _) in warning_mock.mock_calls]
        assert not any(('Consider adding the Results middleware' in x for x in warning_messages))

def test_age_limit_skipped_messages_store_consistent_exceptions(stub_broker, stub_worker, result_backend):
    if False:
        i = 10
        return i + 15
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True, max_age=1)
    def do_work():
        if False:
            while True:
                i = 10
        return 42
    message = do_work.send_with_options(args=[], kwargs={}, delay=2)
    with pytest.raises(ResultFailure) as exc_1:
        result_backend.get_result(message, block=True)
    assert str(exc_1.value) == 'actor raised SkipMessage: Message age limit exceeded'
    assert exc_1.value.orig_exc_type == 'SkipMessage'
    assert exc_1.value.orig_exc_msg == 'Message age limit exceeded'
    time.sleep(0.2)
    with pytest.raises(ResultFailure) as exc_2:
        result_backend.get_result(message)
    assert str(exc_2.value) == str(exc_1.value)
    assert exc_2.value.orig_exc_type == exc_1.value.orig_exc_type
    assert exc_2.value.orig_exc_msg == exc_1.value.orig_exc_msg

def test_custom_skipped_messages_with_no_fail_stores_none(stub_broker, stub_worker, result_backend):
    if False:
        for i in range(10):
            print('nop')
    stub_broker.add_middleware(Results(backend=result_backend))

    class SkipMiddleware(Middleware):

        def before_process_message(self, broker, message):
            if False:
                print('Hello World!')
            raise SkipMessage('Custom skip')
    stub_broker.add_middleware(SkipMiddleware())

    @dramatiq.actor(store_results=True)
    def do_work():
        if False:
            while True:
                i = 10
        return 42
    sent_message = do_work.send()
    assert result_backend.get_result(sent_message, block=True) is None
    time.sleep(0.2)
    assert result_backend.get_result(sent_message) is None