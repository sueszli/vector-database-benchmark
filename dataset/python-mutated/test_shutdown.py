import logging
import time
from unittest import mock
import gevent
import pytest
import dramatiq
from dramatiq.brokers.stub import StubBroker
from dramatiq.middleware import shutdown, threading
from ..common import skip_with_gevent, skip_without_gevent
not_supported = threading.current_platform not in threading.supported_platforms

def test_shutdown_notifications_platform_not_supported(recwarn, monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setattr(shutdown, 'current_platform', 'not supported')
    broker = StubBroker(middleware=[shutdown.ShutdownNotifications()])
    broker.emit_after('process_boot')
    assert len(recwarn) == 1
    assert str(recwarn[0].message) == "ShutdownNotifications cannot kill threads on your current platform ('not supported')."

@skip_with_gevent
@mock.patch('dramatiq.middleware.shutdown.raise_thread_exception')
def test_shutdown_notifications_worker_shutdown_messages(raise_thread_exception, caplog):
    if False:
        return 10
    caplog.set_level(logging.NOTSET)
    middleware = shutdown.ShutdownNotifications()
    middleware.manager.notifications = [1, 2]
    broker = StubBroker(middleware=[middleware])
    broker.emit_before('worker_shutdown', None)
    raise_thread_exception.assert_has_calls([mock.call(1, shutdown.Shutdown), mock.call(2, shutdown.Shutdown)])
    assert len(caplog.record_tuples) == 3
    assert caplog.record_tuples == [('dramatiq.middleware.shutdown.ShutdownNotifications', logging.DEBUG, 'Sending shutdown notification to worker threads...'), ('dramatiq.middleware.shutdown.ShutdownNotifications', logging.INFO, 'Worker shutdown notification. Raising exception in worker thread 1.'), ('dramatiq.middleware.shutdown.ShutdownNotifications', logging.INFO, 'Worker shutdown notification. Raising exception in worker thread 2.')]

@skip_without_gevent
def test_shutdown_notifications_gevent_worker_shutdown_messages(caplog):
    if False:
        print('Hello World!')
    caplog.set_level(logging.NOTSET)
    middleware = shutdown.ShutdownNotifications()
    greenlet_1 = gevent.spawn()
    greenlet_2 = gevent.spawn()
    middleware.manager.notification_greenlets = [(1, greenlet_1), (2, greenlet_2)]
    broker = StubBroker(middleware=[middleware])
    broker.emit_before('worker_shutdown', None)
    assert isinstance(greenlet_1.exception, shutdown.Shutdown)
    assert isinstance(greenlet_2.exception, shutdown.Shutdown)
    assert len(caplog.record_tuples) == 3
    assert caplog.record_tuples == [('dramatiq.middleware.shutdown.ShutdownNotifications', logging.DEBUG, 'Sending shutdown notification to worker threads...'), ('dramatiq.middleware.shutdown.ShutdownNotifications', logging.INFO, 'Worker shutdown notification. Raising exception in worker thread 1.'), ('dramatiq.middleware.shutdown.ShutdownNotifications', logging.INFO, 'Worker shutdown notification. Raising exception in worker thread 2.')]

@pytest.mark.parametrize('actor_opt, message_opt, should_notify', [(True, True, True), (True, False, False), (True, None, True), (False, True, True), (False, False, False), (False, None, False), (None, True, True), (None, False, False), (None, None, False)])
def test_shutdown_notifications_options(stub_broker, actor_opt, message_opt, should_notify):
    if False:
        i = 10
        return i + 15
    middleware = shutdown.ShutdownNotifications()

    @dramatiq.actor(notify_shutdown=actor_opt)
    def do_work():
        if False:
            while True:
                i = 10
        pass
    message = do_work.message_with_options(notify_shutdown=message_opt)
    assert middleware.should_notify(do_work, message) == should_notify

@pytest.mark.skipif(not_supported, reason='Threading not supported on this platform.')
def test_shutdown_notifications_are_received(stub_broker, stub_worker):
    if False:
        return 10
    (shutdowns, successes) = ([], [])

    @dramatiq.actor(notify_shutdown=True, max_retries=0)
    def do_work():
        if False:
            print('Hello World!')
        try:
            for _ in range(10):
                time.sleep(0.1)
        except shutdown.Shutdown:
            shutdowns.append(1)
            raise
        successes.append(1)
    do_work.send()
    time.sleep(0.1)
    stub_worker.stop()
    stub_broker.join(do_work.queue_name)
    stub_worker.join()
    assert sum(shutdowns) == 1
    assert sum(successes) == 0

@pytest.mark.skipif(not_supported, reason='Threading not supported on this platform.')
def test_shutdown_notifications_can_be_ignored(stub_broker, stub_worker):
    if False:
        while True:
            i = 10
    (shutdowns, successes) = ([], [])

    @dramatiq.actor(max_retries=0)
    def do_work():
        if False:
            i = 10
            return i + 15
        try:
            time.sleep(0.2)
        except shutdown.Shutdown:
            shutdowns.append(1)
        else:
            successes.append(1)
    do_work.send()
    time.sleep(0.1)
    stub_worker.stop()
    stub_broker.join(do_work.queue_name)
    stub_worker.join()
    assert sum(shutdowns) == 0
    assert sum(successes) == 1

@pytest.mark.skipif(not_supported, reason='Threading not supported on this platform.')
def test_shutdown_notifications_dont_notify_completed_threads(stub_broker, stub_worker):
    if False:
        return 10
    (shutdowns, successes) = ([], [])

    @dramatiq.actor(notify_shutdown=True, max_retries=0)
    def do_work(n=10, i=0.1):
        if False:
            i = 10
            return i + 15
        try:
            for _ in range(n):
                time.sleep(i)
        except shutdown.Shutdown:
            shutdowns.append(1)
            raise
        successes.append(1)
    do_work.send(n=1)
    do_work.send(n=10)
    time.sleep(0.5)
    stub_worker.stop()
    stub_broker.join(do_work.queue_name)
    stub_worker.join()
    assert sum(shutdowns) == 1
    assert sum(successes) == 1