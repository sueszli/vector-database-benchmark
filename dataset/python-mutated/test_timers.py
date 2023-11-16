import logging
import time
import eventlet
import pytest
from mock import Mock
from nameko.testing.services import entrypoint_hook, get_extension
from nameko.testing.utils import wait_for_call
from nameko.timer import Timer, timer

@pytest.fixture
def tracker():
    if False:
        return 10
    return Mock()

@pytest.mark.parametrize('interval,eager,call_duration,expected_calls', [(0.1, False, 0, 1), (0.1, True, 0, 2), (5, False, 0, 0), (5, True, 0, 1), (0.1, False, 0.3, 1), (0.1, True, 0.3, 1), (0.025, False, 0.07, 2), (0.025, True, 0.07, 3)])
def test_timer_run(interval, eager, call_duration, expected_calls, container_factory, tracker):
    if False:
        i = 10
        return i + 15
    'Test running the timers main loop.\n\n    We test with "timer_only" mode, where only the main loop is run as well as\n    in a more comprehensive mode where the entire container is tested.\n\n    '
    timeout = 0.15
    times = []

    class Service(object):
        name = 'service'

        @timer(interval, eager)
        def tick(self):
            if False:
                return 10
            times.append(time.time())
            eventlet.sleep(call_duration)
            tracker()
    container = container_factory(Service, {})
    instance = get_extension(container, Timer)
    assert instance.interval == interval
    assert instance.eager == eager
    t0 = time.time()
    container.start()
    eventlet.sleep(timeout)
    container.stop()
    rel_times = [t - t0 for t in times]
    assert tracker.call_count == expected_calls, 'Expected {} calls but got {} with {}timer interval of {} and call duration of {}. Times were {}'.format(expected_calls, tracker.call_count, 'eager-' if eager else '', interval, call_duration, rel_times)

def test_kill_stops_timer(container_factory, tracker):
    if False:
        print('Hello World!')

    class Service(object):
        name = 'service'

        @timer(0)
        def tick(self):
            if False:
                return 10
            tracker()
    container = container_factory(Service, {})
    container.start()
    with wait_for_call(1, tracker):
        container.kill()
    eventlet.sleep(0.1)
    assert tracker.call_count == 1

def test_stop_while_sleeping(container_factory, tracker):
    if False:
        print('Hello World!')
    'Check that waiting for the timer to fire does not block the container\n    from being shut down gracefully.\n    '

    class Service(object):
        name = 'service'

        @timer(5)
        def tick(self):
            if False:
                return 10
            tracker()
    container = container_factory(Service, {})
    container.start()
    with eventlet.Timeout(1):
        container.stop()
    assert tracker.call_count == 0, 'Timer should not have fired.'

def test_timer_error(container_factory, caplog, tracker):
    if False:
        return 10
    'Check that an error in the decorated method does not cause the service\n    containers loop to raise an exception.\n    '

    class Service(object):
        name = 'service'

        @timer(5, True)
        def tick(self):
            if False:
                i = 10
                return i + 15
            tracker()
    tracker.side_effect = ValueError('Boom!')
    container = container_factory(Service, {})
    with caplog.at_level(logging.CRITICAL):
        container.start()
        eventlet.sleep(0.05)
        assert tracker.call_count == 1
        container.stop()
    assert len(caplog.records) == 0, 'Expected no errors to have been raised in the worker thread.'

def test_expected_error_in_worker(container_factory, caplog):
    if False:
        print('Hello World!')
    'Make sure that expected exceptions are processed correctly.'

    class ExampleError(Exception):
        pass

    class Service(object):
        name = 'service'

        @timer(1, expected_exceptions=(ExampleError,))
        def tick(self):
            if False:
                return 10
            raise ExampleError('boom!')
    container = container_factory(Service, {})
    with entrypoint_hook(container, 'tick') as tick:
        container.start()
        with pytest.raises(ExampleError):
            tick()
    assert len(caplog.records) == 1
    assert caplog.records[0].message == '(expected) error handling worker {}: boom!'.format(caplog.records[0].args[0])