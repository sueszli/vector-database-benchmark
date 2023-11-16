import os
import time
from threading import Event
from unittest.mock import Mock, patch
import pika.exceptions
import pytest
import dramatiq
from dramatiq import Message, QueueJoinTimeout, Worker
from dramatiq.brokers.rabbitmq import RabbitmqBroker, URLRabbitmqBroker, _IgnoreScaryLogs
from dramatiq.common import current_millis
from .common import RABBITMQ_CREDENTIALS, RABBITMQ_PASSWORD, RABBITMQ_USERNAME

def test_urlrabbitmq_creates_instances_of_rabbitmq_broker():
    if False:
        for i in range(10):
            print('nop')
    url = 'amqp://%s:%s@127.0.0.1:5672' % (RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
    broker = URLRabbitmqBroker(url)
    assert isinstance(broker, RabbitmqBroker)

def test_rabbitmq_broker_can_be_passed_a_semicolon_separated_list_of_uris():
    if False:
        return 10
    broker = RabbitmqBroker(url='amqp://127.0.0.1:55672;amqp://%s:%s@127.0.0.1' % (RABBITMQ_USERNAME, RABBITMQ_PASSWORD))
    assert broker.connection

def test_rabbitmq_broker_can_be_passed_a_list_of_uri_for_failover():
    if False:
        return 10
    broker = RabbitmqBroker(url=['amqp://127.0.0.1:55672', 'amqp://%s:%s@127.0.0.1' % (RABBITMQ_USERNAME, RABBITMQ_PASSWORD)])
    assert broker.connection

def test_rabbitmq_broker_raises_an_error_if_given_invalid_parameter_combinations():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(RuntimeError):
        RabbitmqBroker(url='amqp://127.0.0.1:5672', parameters=[dict(host='127.0.0.1', credentials=RABBITMQ_CREDENTIALS)])
    with pytest.raises(RuntimeError):
        RabbitmqBroker(host='127.0.0.1', url='amqp://127.0.0.1:5672')
    with pytest.raises(RuntimeError):
        RabbitmqBroker(host='127.0.0.1', parameters=[dict(host='127.0.0.1')])

def test_rabbitmq_broker_can_be_passed_a_list_of_parameters_for_failover():
    if False:
        for i in range(10):
            print('nop')
    parameters = [dict(host='127.0.0.1', port=55672), dict(host='127.0.0.1', credentials=RABBITMQ_CREDENTIALS)]
    broker = RabbitmqBroker(parameters=parameters)
    assert broker.connection

def test_rabbitmq_actors_can_be_sent_messages(rabbitmq_broker, rabbitmq_worker):
    if False:
        for i in range(10):
            print('nop')
    database = {}

    @dramatiq.actor
    def put(key, value):
        if False:
            i = 10
            return i + 15
        database[key] = value
    for i in range(100):
        assert put.send('key-%d' % i, i)
    rabbitmq_broker.join(put.queue_name)
    rabbitmq_worker.join()
    assert len(database) == 100

def test_rabbitmq_actors_retry_with_backoff_on_failure(rabbitmq_broker, rabbitmq_worker):
    if False:
        return 10
    (failure_time, success_time) = (None, None)
    succeeded = Event()

    @dramatiq.actor(min_backoff=1000, max_backoff=5000)
    def do_work():
        if False:
            while True:
                i = 10
        nonlocal failure_time, success_time
        if not failure_time:
            failure_time = current_millis()
            raise RuntimeError('First failure.')
        else:
            success_time = current_millis()
            succeeded.set()
    do_work.send()
    succeeded.wait(timeout=30)
    assert 500 <= success_time - failure_time <= 1500

def test_rabbitmq_actors_can_retry_multiple_times(rabbitmq_broker, rabbitmq_worker):
    if False:
        while True:
            i = 10
    attempts = []

    @dramatiq.actor(max_backoff=1000)
    def do_work():
        if False:
            while True:
                i = 10
        attempts.append(1)
        if sum(attempts) < 4:
            raise RuntimeError('Failure #%d' % sum(attempts))
    do_work.send()
    rabbitmq_broker.join(do_work.queue_name, min_successes=40)
    rabbitmq_worker.join()
    assert sum(attempts) == 4

def test_rabbitmq_actors_can_have_their_messages_delayed(rabbitmq_broker, rabbitmq_worker):
    if False:
        i = 10
        return i + 15
    (start_time, run_time) = (current_millis(), None)

    @dramatiq.actor
    def record():
        if False:
            print('Hello World!')
        nonlocal run_time
        run_time = current_millis()
    record.send_with_options(delay=1000)
    rabbitmq_broker.join(record.queue_name)
    rabbitmq_worker.join()
    assert run_time - start_time >= 1000

def test_rabbitmq_actors_can_delay_messages_independent_of_each_other(rabbitmq_broker):
    if False:
        print('Hello World!')
    results = []

    @dramatiq.actor
    def append(x):
        if False:
            for i in range(10):
                print('nop')
        results.append(x)
    broker = rabbitmq_broker
    worker = Worker(broker, worker_threads=1)
    try:
        append.send_with_options(args=(1,), delay=1500)
        append.send_with_options(args=(2,), delay=1000)
        worker.start()
        broker.join(append.queue_name, min_successes=20)
        worker.join()
        assert results == [2, 1]
    finally:
        worker.stop()

def test_rabbitmq_actors_can_have_retry_limits(rabbitmq_broker, rabbitmq_worker):
    if False:
        for i in range(10):
            print('nop')

    @dramatiq.actor(max_retries=0)
    def do_work():
        if False:
            while True:
                i = 10
        raise RuntimeError('failed')
    do_work.send()
    rabbitmq_broker.join(do_work.queue_name)
    rabbitmq_worker.join()
    (_, _, xq_count) = rabbitmq_broker.get_queue_message_counts(do_work.queue_name)
    assert xq_count == 1

def test_rabbitmq_broker_connections_are_lazy():
    if False:
        print('Hello World!')
    broker = RabbitmqBroker(host='127.0.0.1', max_priority=10, credentials=RABBITMQ_CREDENTIALS)

    def get_connection():
        if False:
            print('Hello World!')
        return getattr(broker.state, 'connection', None)
    assert get_connection() is None
    broker.declare_queue('some-queue')
    assert get_connection() is None
    broker.consume('some-queue', timeout=1)
    assert get_connection() is not None

def test_rabbitmq_broker_stops_retrying_declaring_queues_when_max_attempts_reached(rabbitmq_broker):
    if False:
        i = 10
        return i + 15
    with patch.object(rabbitmq_broker, '_declare_queue', side_effect=pika.exceptions.AMQPConnectionError):
        with pytest.raises(dramatiq.errors.ConnectionClosed):

            @dramatiq.actor(queue_name='flaky_queue')
            def do_work():
                if False:
                    i = 10
                    return i + 15
                pass
            do_work.send()

def test_rabbitmq_messages_belonging_to_missing_actors_are_rejected(rabbitmq_broker, rabbitmq_worker):
    if False:
        i = 10
        return i + 15
    message = Message(queue_name='some-queue', actor_name='some-actor', args=(), kwargs={}, options={})
    rabbitmq_broker.declare_queue(message.queue_name)
    rabbitmq_broker.enqueue(message)
    rabbitmq_broker.join(message.queue_name)
    rabbitmq_worker.join()
    (_, _, dead) = rabbitmq_broker.get_queue_message_counts(message.queue_name)
    assert dead == 1

def test_rabbitmq_broker_reconnects_after_enqueue_failure(rabbitmq_broker):
    if False:
        while True:
            i = 10

    @dramatiq.actor
    def do_nothing():
        if False:
            print('Hello World!')
        pass
    rabbitmq_broker.connection.close()
    assert do_nothing.send()
    assert rabbitmq_broker.connection.is_open

def test_rabbitmq_workers_handle_rabbit_failures_gracefully(rabbitmq_broker, rabbitmq_worker):
    if False:
        for i in range(10):
            print('nop')
    attempts = []

    @dramatiq.actor
    def do_work():
        if False:
            i = 10
            return i + 15
        attempts.append(1)
        time.sleep(1)
    do_work.send_with_options(delay=1000)
    os.system('rabbitmqctl stop_app')
    os.system('rabbitmqctl start_app')
    del rabbitmq_broker.channel
    del rabbitmq_broker.connection
    rabbitmq_broker.join(do_work.queue_name)
    rabbitmq_worker.join()
    assert sum(attempts) >= 1

def test_rabbitmq_connections_can_be_deleted_multiple_times(rabbitmq_broker):
    if False:
        return 10
    del rabbitmq_broker.connection
    del rabbitmq_broker.connection

def test_rabbitmq_channels_can_be_deleted_multiple_times(rabbitmq_broker):
    if False:
        print('Hello World!')
    del rabbitmq_broker.channel
    del rabbitmq_broker.channel

def test_rabbitmq_consumers_ignore_unknown_messages_in_ack_and_nack(rabbitmq_broker):
    if False:
        for i in range(10):
            print('nop')
    consumer = rabbitmq_broker.consume('default')
    assert consumer.ack(Mock(_tag=1)) is None
    assert consumer.nack(Mock(_tag=1)) is None

def test_ignore_scary_logs_filter_ignores_logs():
    if False:
        print('Hello World!')
    log_filter = _IgnoreScaryLogs('pika.adapters')
    record = Mock()
    record.getMessage.return_value = "ConnectionError('Broken pipe')"
    assert not log_filter.filter(record)
    record = Mock()
    record.getMessage.return_value = 'Not scary'
    assert log_filter.filter(record)

def test_rabbitmq_broker_can_join_with_timeout(rabbitmq_broker, rabbitmq_worker):
    if False:
        while True:
            i = 10

    @dramatiq.actor
    def do_work():
        if False:
            print('Hello World!')
        time.sleep(1)
    do_work.send()
    with pytest.raises(QueueJoinTimeout):
        rabbitmq_broker.join(do_work.queue_name, timeout=500)

def test_rabbitmq_broker_can_flush_queues(rabbitmq_broker):
    if False:
        print('Hello World!')

    @dramatiq.actor
    def do_work():
        if False:
            return 10
        pass
    do_work.send()
    rabbitmq_broker.flush_all()
    assert rabbitmq_broker.join(do_work.queue_name, min_successes=1, timeout=200) is None

def test_rabbitmq_broker_can_enqueue_messages_with_priority(rabbitmq_broker):
    if False:
        i = 10
        return i + 15
    max_priority = 10
    message_processing_order = []
    queue_name = 'prioritized'

    @dramatiq.actor(queue_name=queue_name)
    def do_work(message_priority):
        if False:
            return 10
        message_processing_order.append(message_priority)
    worker = Worker(rabbitmq_broker, worker_threads=1)
    worker.queue_prefetch = 1
    worker.start()
    worker.pause()
    try:
        for priority in range(max_priority):
            do_work.send_with_options(args=(priority,), broker_priority=priority)
        worker.resume()
        rabbitmq_broker.join(queue_name, timeout=5000)
        worker.join()
        assert message_processing_order == list(reversed(range(max_priority)))
    finally:
        worker.stop()

def test_rabbitmq_broker_retries_declaring_queues_when_connection_related_errors_occur(rabbitmq_broker):
    if False:
        print('Hello World!')
    (executed, declare_called) = (False, False)
    original_declare = rabbitmq_broker._declare_queue

    def flaky_declare_queue(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        nonlocal declare_called
        if not declare_called:
            declare_called = True
            raise pika.exceptions.AMQPConnectionError
        return original_declare(*args, **kwargs)
    with patch.object(rabbitmq_broker, '_declare_queue', flaky_declare_queue):

        @dramatiq.actor(queue_name='flaky_queue')
        def do_work():
            if False:
                while True:
                    i = 10
            nonlocal executed
            executed = True
        do_work.send()
        worker = Worker(rabbitmq_broker, worker_threads=1)
        worker.start()
        try:
            rabbitmq_broker.join(do_work.queue_name, timeout=5000)
            worker.join()
            assert declare_called
            assert executed
        finally:
            worker.stop()

def test_rabbitmq_messages_that_failed_to_decode_are_rejected(rabbitmq_broker, rabbitmq_worker):
    if False:
        print('Hello World!')

    @dramatiq.actor(max_retries=0)
    def do_work(_):
        if False:
            for i in range(10):
                print('nop')
        pass
    old_encoder = dramatiq.get_encoder()

    class BadEncoder(type(old_encoder)):

        def decode(self, data):
            if False:
                print('Hello World!')
            if 'xfail' in str(data):
                raise RuntimeError('xfail')
            return super().decode(data)
    dramatiq.set_encoder(BadEncoder())
    try:
        do_work.send('xfail')
        rabbitmq_broker.join(do_work.queue_name)
        rabbitmq_worker.join()
        (q_count, dq_count, xq_count) = rabbitmq_broker.get_queue_message_counts(do_work.queue_name)
        assert q_count == dq_count == 0
        assert xq_count == 1
    finally:
        dramatiq.set_encoder(old_encoder)

def test_rabbitmq_queues_only_contains_canonical_name(rabbitmq_broker, rabbitmq_worker):
    if False:
        for i in range(10):
            print('nop')
    assert len(rabbitmq_broker.queues) == 0

    @dramatiq.actor
    def put():
        if False:
            i = 10
            return i + 15
        pass
    assert len(rabbitmq_broker.queues) == 1
    assert put.queue_name in rabbitmq_broker.queues