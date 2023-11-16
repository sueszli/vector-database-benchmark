import logging
import time
from unittest import mock
import pytest
from greenlet import getcurrent
import dramatiq
from dramatiq.brokers.stub import StubBroker
from dramatiq.middleware import threading, time_limit
from ..common import skip_with_gevent, skip_without_gevent
not_supported = threading.current_platform not in threading.supported_platforms

@skip_with_gevent
def test_time_limit_platform_not_supported(recwarn, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(time_limit, 'current_platform', 'not supported')
    broker = StubBroker(middleware=[time_limit.TimeLimit()])
    broker.emit_after('process_boot')
    assert len(recwarn) == 1
    assert str(recwarn[0].message) == "TimeLimit cannot kill threads on your current platform ('not supported')."

@skip_with_gevent
@mock.patch('dramatiq.middleware.time_limit.raise_thread_exception')
def test_time_limit_exceeded_worker_messages(raise_thread_exception, caplog):
    if False:
        i = 10
        return i + 15
    caplog.set_level(logging.NOTSET)
    current_time = time.monotonic()
    middleware = time_limit.TimeLimit()
    middleware.manager.deadlines = {1: current_time - 2, 2: current_time - 1, 3: current_time + 50000}
    middleware.manager._handle()
    raise_thread_exception.assert_has_calls([mock.call(1, time_limit.TimeLimitExceeded), mock.call(2, time_limit.TimeLimitExceeded)])
    assert len(caplog.record_tuples) == 2
    assert caplog.record_tuples == [('dramatiq.middleware.time_limit.TimeLimit', logging.WARNING, 'Time limit exceeded. Raising exception in worker thread 1.'), ('dramatiq.middleware.time_limit.TimeLimit', logging.WARNING, 'Time limit exceeded. Raising exception in worker thread 2.')]

@skip_without_gevent
@mock.patch('dramatiq.middleware.time_limit.Timeout._on_expiration')
def test_time_limit_exceeded_gevent_worker_messages(on_expiration, caplog):
    if False:
        while True:
            i = 10
    caplog.set_level(logging.NOTSET)
    middleware = time_limit.TimeLimit()
    timer_1 = time_limit._GeventTimeout(seconds=0.01, thread_id=1, logger=middleware.logger, exception=time_limit.TimeLimitExceeded)
    timer_2 = time_limit._GeventTimeout(seconds=0.02, thread_id=2, logger=middleware.logger, exception=time_limit.TimeLimitExceeded)
    timer_3 = time_limit._GeventTimeout(seconds=10, thread_id=3, logger=middleware.logger, exception=time_limit.TimeLimitExceeded)
    timer_1.start()
    timer_2.start()
    timer_3.start()
    time.sleep(0.1)
    current_greenlet = getcurrent()
    assert on_expiration.assert_has_calls
    on_expiration.assert_has_calls([mock.call(current_greenlet, time_limit.TimeLimitExceeded), mock.call(current_greenlet, time_limit.TimeLimitExceeded)])
    assert len(caplog.record_tuples) == 2
    assert caplog.record_tuples == [('dramatiq.middleware.time_limit.TimeLimit', logging.WARNING, 'Time limit exceeded. Raising exception in worker thread 1.'), ('dramatiq.middleware.time_limit.TimeLimit', logging.WARNING, 'Time limit exceeded. Raising exception in worker thread 2.')]
    timer_1.close()
    timer_2.close()
    timer_3.close()

@pytest.mark.skipif(not_supported, reason='Threading not supported on this platform.')
def test_time_limits_are_handled(stub_broker, stub_worker):
    if False:
        return 10
    (time_limits_exceeded, successes) = ([], [])

    @dramatiq.actor(time_limit=10, max_retries=0)
    def do_work():
        if False:
            for i in range(10):
                print('nop')
        try:
            for _ in range(20):
                time.sleep(0.1)
        except time_limit.TimeLimitExceeded:
            time_limits_exceeded.append(1)
            raise
        successes.append(1)
    do_work.send()
    stub_broker.join(do_work.queue_name)
    stub_worker.join()
    assert sum(time_limits_exceeded) == 1
    assert sum(successes) == 0