"""Async I/O backend support utilities."""
import socket
import threading
import time
from collections import deque
from queue import Empty
from time import sleep
from weakref import WeakKeyDictionary
from kombu.utils.compat import detect_environment
from celery import states
from celery.exceptions import TimeoutError
from celery.utils.threads import THREAD_TIMEOUT_MAX
__all__ = ('AsyncBackendMixin', 'BaseResultConsumer', 'Drainer', 'register_drainer')
drainers = {}

def register_drainer(name):
    if False:
        i = 10
        return i + 15
    'Decorator used to register a new result drainer type.'

    def _inner(cls):
        if False:
            for i in range(10):
                print('nop')
        drainers[name] = cls
        return cls
    return _inner

@register_drainer('default')
class Drainer:
    """Result draining service."""

    def __init__(self, result_consumer):
        if False:
            for i in range(10):
                print('nop')
        self.result_consumer = result_consumer

    def start(self):
        if False:
            i = 10
            return i + 15
        pass

    def stop(self):
        if False:
            while True:
                i = 10
        pass

    def drain_events_until(self, p, timeout=None, interval=1, on_interval=None, wait=None):
        if False:
            i = 10
            return i + 15
        wait = wait or self.result_consumer.drain_events
        time_start = time.monotonic()
        while 1:
            if timeout and time.monotonic() - time_start >= timeout:
                raise socket.timeout()
            try:
                yield self.wait_for(p, wait, timeout=interval)
            except socket.timeout:
                pass
            if on_interval:
                on_interval()
            if p.ready:
                break

    def wait_for(self, p, wait, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        wait(timeout=timeout)

class greenletDrainer(Drainer):
    spawn = None
    _g = None
    _drain_complete_event = None

    def _create_drain_complete_event(self):
        if False:
            return 10
        'create new self._drain_complete_event object'
        pass

    def _send_drain_complete_event(self):
        if False:
            while True:
                i = 10
        'raise self._drain_complete_event for wakeup .wait_for'
        pass

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._shutdown = threading.Event()
        self._create_drain_complete_event()

    def run(self):
        if False:
            return 10
        self._started.set()
        while not self._stopped.is_set():
            try:
                self.result_consumer.drain_events(timeout=1)
                self._send_drain_complete_event()
                self._create_drain_complete_event()
            except socket.timeout:
                pass
        self._shutdown.set()

    def start(self):
        if False:
            print('Hello World!')
        if not self._started.is_set():
            self._g = self.spawn(self.run)
            self._started.wait()

    def stop(self):
        if False:
            print('Hello World!')
        self._stopped.set()
        self._send_drain_complete_event()
        self._shutdown.wait(THREAD_TIMEOUT_MAX)

    def wait_for(self, p, wait, timeout=None):
        if False:
            i = 10
            return i + 15
        self.start()
        if not p.ready:
            self._drain_complete_event.wait(timeout=timeout)

@register_drainer('eventlet')
class eventletDrainer(greenletDrainer):

    def spawn(self, func):
        if False:
            print('Hello World!')
        from eventlet import sleep, spawn
        g = spawn(func)
        sleep(0)
        return g

    def _create_drain_complete_event(self):
        if False:
            return 10
        from eventlet.event import Event
        self._drain_complete_event = Event()

    def _send_drain_complete_event(self):
        if False:
            return 10
        self._drain_complete_event.send()

@register_drainer('gevent')
class geventDrainer(greenletDrainer):

    def spawn(self, func):
        if False:
            i = 10
            return i + 15
        import gevent
        g = gevent.spawn(func)
        gevent.sleep(0)
        return g

    def _create_drain_complete_event(self):
        if False:
            while True:
                i = 10
        from gevent.event import Event
        self._drain_complete_event = Event()

    def _send_drain_complete_event(self):
        if False:
            i = 10
            return i + 15
        self._drain_complete_event.set()
        self._create_drain_complete_event()

class AsyncBackendMixin:
    """Mixin for backends that enables the async API."""

    def _collect_into(self, result, bucket):
        if False:
            return 10
        self.result_consumer.buckets[result] = bucket

    def iter_native(self, result, no_ack=True, **kwargs):
        if False:
            print('Hello World!')
        self._ensure_not_eager()
        results = result.results
        if not results:
            raise StopIteration()
        bucket = deque()
        for node in results:
            if not hasattr(node, '_cache'):
                bucket.append(node)
            elif node._cache:
                bucket.append(node)
            else:
                self._collect_into(node, bucket)
        for _ in self._wait_for_pending(result, no_ack=no_ack, **kwargs):
            while bucket:
                node = bucket.popleft()
                if not hasattr(node, '_cache'):
                    yield (node.id, node.children)
                else:
                    yield (node.id, node._cache)
        while bucket:
            node = bucket.popleft()
            yield (node.id, node._cache)

    def add_pending_result(self, result, weak=False, start_drainer=True):
        if False:
            i = 10
            return i + 15
        if start_drainer:
            self.result_consumer.drainer.start()
        try:
            self._maybe_resolve_from_buffer(result)
        except Empty:
            self._add_pending_result(result.id, result, weak=weak)
        return result

    def _maybe_resolve_from_buffer(self, result):
        if False:
            print('Hello World!')
        result._maybe_set_cache(self._pending_messages.take(result.id))

    def _add_pending_result(self, task_id, result, weak=False):
        if False:
            for i in range(10):
                print('nop')
        (concrete, weak_) = self._pending_results
        if task_id not in weak_ and result.id not in concrete:
            (weak_ if weak else concrete)[task_id] = result
            self.result_consumer.consume_from(task_id)

    def add_pending_results(self, results, weak=False):
        if False:
            while True:
                i = 10
        self.result_consumer.drainer.start()
        return [self.add_pending_result(result, weak=weak, start_drainer=False) for result in results]

    def remove_pending_result(self, result):
        if False:
            return 10
        self._remove_pending_result(result.id)
        self.on_result_fulfilled(result)
        return result

    def _remove_pending_result(self, task_id):
        if False:
            print('Hello World!')
        for mapping in self._pending_results:
            mapping.pop(task_id, None)

    def on_result_fulfilled(self, result):
        if False:
            return 10
        self.result_consumer.cancel_for(result.id)

    def wait_for_pending(self, result, callback=None, propagate=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._ensure_not_eager()
        for _ in self._wait_for_pending(result, **kwargs):
            pass
        return result.maybe_throw(callback=callback, propagate=propagate)

    def _wait_for_pending(self, result, timeout=None, on_interval=None, on_message=None, **kwargs):
        if False:
            print('Hello World!')
        return self.result_consumer._wait_for_pending(result, timeout=timeout, on_interval=on_interval, on_message=on_message, **kwargs)

    @property
    def is_async(self):
        if False:
            while True:
                i = 10
        return True

class BaseResultConsumer:
    """Manager responsible for consuming result messages."""

    def __init__(self, backend, app, accept, pending_results, pending_messages):
        if False:
            for i in range(10):
                print('nop')
        self.backend = backend
        self.app = app
        self.accept = accept
        self._pending_results = pending_results
        self._pending_messages = pending_messages
        self.on_message = None
        self.buckets = WeakKeyDictionary()
        self.drainer = drainers[detect_environment()](self)

    def start(self, initial_task_id, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def stop(self):
        if False:
            print('Hello World!')
        pass

    def drain_events(self, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def consume_from(self, task_id):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def cancel_for(self, task_id):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def _after_fork(self):
        if False:
            i = 10
            return i + 15
        self.buckets.clear()
        self.buckets = WeakKeyDictionary()
        self.on_message = None
        self.on_after_fork()

    def on_after_fork(self):
        if False:
            while True:
                i = 10
        pass

    def drain_events_until(self, p, timeout=None, on_interval=None):
        if False:
            print('Hello World!')
        return self.drainer.drain_events_until(p, timeout=timeout, on_interval=on_interval)

    def _wait_for_pending(self, result, timeout=None, on_interval=None, on_message=None, **kwargs):
        if False:
            print('Hello World!')
        self.on_wait_for_pending(result, timeout=timeout, **kwargs)
        (prev_on_m, self.on_message) = (self.on_message, on_message)
        try:
            for _ in self.drain_events_until(result.on_ready, timeout=timeout, on_interval=on_interval):
                yield
                sleep(0)
        except socket.timeout:
            raise TimeoutError('The operation timed out.')
        finally:
            self.on_message = prev_on_m

    def on_wait_for_pending(self, result, timeout=None, **kwargs):
        if False:
            print('Hello World!')
        pass

    def on_out_of_band_result(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.on_state_change(message.payload, message)

    def _get_pending_result(self, task_id):
        if False:
            i = 10
            return i + 15
        for mapping in self._pending_results:
            try:
                return mapping[task_id]
            except KeyError:
                pass
        raise KeyError(task_id)

    def on_state_change(self, meta, message):
        if False:
            i = 10
            return i + 15
        if self.on_message:
            self.on_message(meta)
        if meta['status'] in states.READY_STATES:
            task_id = meta['task_id']
            try:
                result = self._get_pending_result(task_id)
            except KeyError:
                self._pending_messages.put(task_id, meta)
            else:
                result._maybe_set_cache(meta)
                buckets = self.buckets
                try:
                    bucket = buckets.pop(result)
                except KeyError:
                    pass
                else:
                    bucket.append(result)
        sleep(0)