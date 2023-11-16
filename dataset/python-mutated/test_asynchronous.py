import os
import socket
import sys
import threading
import time
from unittest.mock import Mock, patch
import pytest
from vine import promise
from celery.backends.asynchronous import BaseResultConsumer
from celery.backends.base import Backend
from celery.utils import cached_property
pytest.importorskip('gevent')
pytest.importorskip('eventlet')

@pytest.fixture(autouse=True)
def setup_eventlet():
    if False:
        i = 10
        return i + 15
    os.environ.update(EVENTLET_NO_GREENDNS='yes')

class DrainerTests:
    """
    Base test class for the Default / Gevent / Eventlet drainers.
    """
    interval = 0.1
    MAX_TIMEOUT = 10

    def get_drainer(self, environment):
        if False:
            while True:
                i = 10
        with patch('celery.backends.asynchronous.detect_environment') as d:
            d.return_value = environment
            backend = Backend(self.app)
            consumer = BaseResultConsumer(backend, self.app, backend.accept, pending_results={}, pending_messages={})
            consumer.drain_events = Mock(side_effect=self.result_consumer_drain_events)
            return consumer.drainer

    @pytest.fixture(autouse=True)
    def setup_drainer(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @cached_property
    def sleep(self):
        if False:
            i = 10
            return i + 15
        '\n        Sleep on the event loop.\n        '
        raise NotImplementedError

    def schedule_thread(self, thread):
        if False:
            while True:
                i = 10
        '\n        Set up a thread that runs on the event loop.\n        '
        raise NotImplementedError

    def teardown_thread(self, thread):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wait for a thread to stop.\n        '
        raise NotImplementedError

    def result_consumer_drain_events(self, timeout=None):
        if False:
            while True:
                i = 10
        '\n        Subclasses should override this method to define the behavior of\n        drainer.result_consumer.drain_events.\n        '
        raise NotImplementedError

    def test_drain_checks_on_interval(self):
        if False:
            i = 10
            return i + 15
        p = promise()

        def fulfill_promise_thread():
            if False:
                while True:
                    i = 10
            self.sleep(self.interval * 2)
            p('done')
        fulfill_thread = self.schedule_thread(fulfill_promise_thread)
        on_interval = Mock()
        for _ in self.drainer.drain_events_until(p, on_interval=on_interval, interval=self.interval, timeout=self.MAX_TIMEOUT):
            pass
        self.teardown_thread(fulfill_thread)
        assert p.ready, 'Should have terminated with promise being ready'
        assert on_interval.call_count < 20, 'Should have limited number of calls to on_interval'

    def test_drain_does_not_block_event_loop(self):
        if False:
            return 10
        '\n        This test makes sure that other greenlets can still operate while drain_events_until is\n        running.\n        '
        p = promise()
        liveness_mock = Mock()

        def fulfill_promise_thread():
            if False:
                i = 10
                return i + 15
            self.sleep(self.interval * 2)
            p('done')

        def liveness_thread():
            if False:
                for i in range(10):
                    print('nop')
            while 1:
                if p.ready:
                    return
                self.sleep(self.interval / 10)
                liveness_mock()
        fulfill_thread = self.schedule_thread(fulfill_promise_thread)
        liveness_thread = self.schedule_thread(liveness_thread)
        on_interval = Mock()
        for _ in self.drainer.drain_events_until(p, on_interval=on_interval, interval=self.interval, timeout=self.MAX_TIMEOUT):
            pass
        self.teardown_thread(fulfill_thread)
        self.teardown_thread(liveness_thread)
        assert p.ready, 'Should have terminated with promise being ready'
        assert on_interval.call_count <= liveness_mock.call_count, 'Should have served liveness_mock while waiting for event'

    def test_drain_timeout(self):
        if False:
            i = 10
            return i + 15
        p = promise()
        on_interval = Mock()
        with pytest.raises(socket.timeout):
            for _ in self.drainer.drain_events_until(p, on_interval=on_interval, interval=self.interval, timeout=self.interval * 5):
                pass
        assert not p.ready, 'Promise should remain un-fulfilled'
        assert on_interval.call_count < 20, 'Should have limited number of calls to on_interval'

@pytest.mark.skipif(sys.platform == 'win32', reason='hangs forever intermittently on windows')
class test_EventletDrainer(DrainerTests):

    @pytest.fixture(autouse=True)
    def setup_drainer(self):
        if False:
            return 10
        self.drainer = self.get_drainer('eventlet')

    @cached_property
    def sleep(self):
        if False:
            return 10
        from eventlet import sleep
        return sleep

    def result_consumer_drain_events(self, timeout=None):
        if False:
            while True:
                i = 10
        import eventlet
        eventlet.sleep(timeout / 10)

    def schedule_thread(self, thread):
        if False:
            i = 10
            return i + 15
        import eventlet
        g = eventlet.spawn(thread)
        eventlet.sleep(0)
        return g

    def teardown_thread(self, thread):
        if False:
            i = 10
            return i + 15
        thread.wait()

class test_Drainer(DrainerTests):

    @pytest.fixture(autouse=True)
    def setup_drainer(self):
        if False:
            for i in range(10):
                print('nop')
        self.drainer = self.get_drainer('default')

    @cached_property
    def sleep(self):
        if False:
            print('Hello World!')
        from time import sleep
        return sleep

    def result_consumer_drain_events(self, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        time.sleep(timeout)

    def schedule_thread(self, thread):
        if False:
            for i in range(10):
                print('nop')
        t = threading.Thread(target=thread)
        t.start()
        return t

    def teardown_thread(self, thread):
        if False:
            while True:
                i = 10
        thread.join()

class test_GeventDrainer(DrainerTests):

    @pytest.fixture(autouse=True)
    def setup_drainer(self):
        if False:
            return 10
        self.drainer = self.get_drainer('gevent')

    @cached_property
    def sleep(self):
        if False:
            for i in range(10):
                print('nop')
        from gevent import sleep
        return sleep

    def result_consumer_drain_events(self, timeout=None):
        if False:
            while True:
                i = 10
        import gevent
        gevent.sleep(timeout / 10)

    def schedule_thread(self, thread):
        if False:
            for i in range(10):
                print('nop')
        import gevent
        g = gevent.spawn(thread)
        gevent.sleep(0)
        return g

    def teardown_thread(self, thread):
        if False:
            while True:
                i = 10
        import gevent
        gevent.wait([thread])