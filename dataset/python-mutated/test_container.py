import sys
from functools import partial
import eventlet
import greenlet
import pytest
from eventlet import Timeout, sleep, spawn
from eventlet.event import Event
from mock import ANY, Mock, call, patch
from nameko.constants import MAX_WORKERS_CONFIG_KEY
from nameko.containers import ServiceContainer, get_service_name
from nameko.exceptions import ConfigurationError
from nameko.extensions import DependencyProvider, Entrypoint
from nameko.testing.utils import get_extension

class CallCollectorMixin(object):
    call_counter = 0

    def __init__(self):
        if False:
            print('Hello World!')
        self._reset_calls()

    def bind(self, *args):
        if False:
            i = 10
            return i + 15
        res = super(CallCollectorMixin, self).bind(*args)
        self.instances.add(res)
        return res

    def _reset_calls(self):
        if False:
            return 10
        self.calls = []
        self.call_ids = []

    def _log_call(self, data):
        if False:
            return 10
        CallCollectorMixin.call_counter += 1
        self.calls.append(data)
        self.call_ids.append(CallCollectorMixin.call_counter)

    def setup(self):
        if False:
            return 10
        self._log_call('setup')
        super(CallCollectorMixin, self).setup()

    def start(self):
        if False:
            i = 10
            return i + 15
        self._log_call('start')
        super(CallCollectorMixin, self).start()

    def stop(self):
        if False:
            return 10
        self._log_call('stop')
        super(CallCollectorMixin, self).stop()

    def kill(self, exc=None):
        if False:
            print('Hello World!')
        self._log_call('kill')
        super(CallCollectorMixin, self).stop()

class CallCollectingEntrypoint(CallCollectorMixin, Entrypoint):
    instances = set()
    expected_exceptions = ()

class CallCollectingDependencyProvider(CallCollectorMixin, DependencyProvider):
    instances = set()

    def get_dependency(self, worker_ctx):
        if False:
            i = 10
            return i + 15
        self._log_call(('get_dependency', worker_ctx))
        return 'spam-attr'

    def worker_setup(self, worker_ctx):
        if False:
            i = 10
            return i + 15
        self._log_call(('worker_setup', worker_ctx))
        super(CallCollectorMixin, self).worker_setup(worker_ctx)

    def worker_result(self, worker_ctx, result=None, exc_info=None):
        if False:
            return 10
        self._log_call(('worker_result', worker_ctx, (result, exc_info)))
        super(CallCollectorMixin, self).worker_result(worker_ctx, result, exc_info)

    def worker_teardown(self, worker_ctx):
        if False:
            while True:
                i = 10
        self._log_call(('worker_teardown', worker_ctx))
        super(CallCollectorMixin, self).worker_teardown(worker_ctx)
foobar = CallCollectingEntrypoint.decorator
egg_error = Exception('broken')

class Service(object):
    name = 'test-service'
    spam = CallCollectingDependencyProvider()

    @foobar
    def ham(self):
        if False:
            while True:
                i = 10
        return 'ham'

    @foobar
    def egg(self):
        if False:
            return 10
        raise egg_error

@pytest.fixture
def container():
    if False:
        return 10
    container = ServiceContainer(Service, config={})
    for ext in container.extensions:
        ext._reset_calls()
    CallCollectorMixin.call_counter = 0
    return container

@pytest.fixture
def logger():
    if False:
        print('Hello World!')
    with patch('nameko.containers._log', autospec=True) as patched:
        yield patched

def test_collects_extensions(container):
    if False:
        for i in range(10):
            print('nop')
    assert len(container.extensions) == 3
    assert container.extensions == CallCollectingEntrypoint.instances | CallCollectingDependencyProvider.instances

def test_starts_extensions(container):
    if False:
        i = 10
        return i + 15
    for ext in container.extensions:
        assert ext.calls == []
    container.start()
    for ext in container.extensions:
        assert ext.calls == ['setup', 'start']

def test_stops_extensions(container):
    if False:
        for i in range(10):
            print('nop')
    container.stop()
    for ext in container.extensions:
        assert ext.calls == ['stop']

def test_stops_entrypoints_before_dependency_providers(container):
    if False:
        for i in range(10):
            print('nop')
    container.stop()
    provider = get_extension(container, DependencyProvider)
    for entrypoint in container.entrypoints:
        assert entrypoint.call_ids[0] < provider.call_ids[0]

def test_worker_life_cycle(container):
    if False:
        i = 10
        return i + 15
    spam_dep = get_extension(container, DependencyProvider)
    ham_dep = get_extension(container, Entrypoint, method_name='ham')
    egg_dep = get_extension(container, Entrypoint, method_name='egg')
    handle_result = Mock()
    handle_result.side_effect = lambda worker_ctx, res, exc_info: (res, exc_info)
    ham_worker_ctx = container.spawn_worker(ham_dep, [], {}, handle_result=handle_result)
    container._worker_pool.waitall()
    egg_worker_ctx = container.spawn_worker(egg_dep, [], {}, handle_result=handle_result)
    container._worker_pool.waitall()
    assert spam_dep.calls == [('get_dependency', ham_worker_ctx), ('worker_setup', ham_worker_ctx), ('worker_result', ham_worker_ctx, ('ham', None)), ('worker_teardown', ham_worker_ctx), ('get_dependency', egg_worker_ctx), ('worker_setup', egg_worker_ctx), ('worker_result', egg_worker_ctx, (None, (Exception, egg_error, ANY))), ('worker_teardown', egg_worker_ctx)]
    assert handle_result.call_args_list == [call(ham_worker_ctx, 'ham', None), call(egg_worker_ctx, None, (Exception, egg_error, ANY))]

def test_wait_waits_for_container_stopped(container):
    if False:
        while True:
            i = 10
    gt = spawn(container.wait)
    with Timeout(1):
        assert not gt.dead
        container.stop()
        sleep(0.01)
        assert gt.dead

def test_container_doesnt_exhaust_max_workers(container):
    if False:
        for i in range(10):
            print('nop')
    spam_called = Event()
    spam_continue = Event()

    class Service(object):
        name = 'max-workers'

        @foobar
        def spam(self, a):
            if False:
                print('Hello World!')
            spam_called.send(a)
            spam_continue.wait()
    container = ServiceContainer(Service, config={MAX_WORKERS_CONFIG_KEY: 1})
    dep = get_extension(container, Entrypoint)
    container.spawn_worker(dep, ['ham'], {})
    gt = spawn(container.spawn_worker, dep, ['eggs'], {})
    with Timeout(1):
        assert spam_called.wait() == 'ham'
        assert not gt.dead
        spam_called.reset()
        spam_continue.send(None)
        assert spam_called.wait() == 'eggs'
        assert gt.dead

def test_stop_already_stopped(container, logger):
    if False:
        for i in range(10):
            print('nop')
    assert not container._died.ready()
    container.stop()
    assert container._died.ready()
    container.stop()
    assert logger.debug.call_args == call('already stopped %s', container)

def test_kill_already_stopped(container, logger):
    if False:
        for i in range(10):
            print('nop')
    assert not container._died.ready()
    container.stop()
    assert container._died.ready()
    container.kill()
    assert logger.debug.call_args == call('already stopped %s', container)

def test_kill_container_with_managed_threads(container):
    if False:
        while True:
            i = 10
    " Start a thread that's not managed by dependencies. Ensure it is killed\n    when the container is.\n    "

    def sleep_forever():
        if False:
            i = 10
            return i + 15
        while True:
            sleep()
    container.spawn_managed_thread(sleep_forever)
    assert len(container._managed_threads) == 1
    (worker_gt,) = container._managed_threads.keys()
    container.kill()
    with Timeout(1):
        container._died.wait()
        with pytest.raises(greenlet.GreenletExit):
            worker_gt.wait()

def test_kill_container_with_active_workers(container_factory):
    if False:
        return 10
    waiting = Event()
    wait_forever = Event()

    class Service(object):
        name = 'kill-with-active-workers'

        @foobar
        def spam(self):
            if False:
                return 10
            waiting.send(None)
            wait_forever.wait()
    container = container_factory(Service, {})
    dep = get_extension(container, Entrypoint)
    container.spawn_worker(dep, (), {})
    waiting.wait()
    with patch('nameko.containers._log') as logger:
        container.kill()
    assert logger.warning.call_args_list == [call('killing %s active workers(s)', 1), call('killing active worker for %s', ANY)]

def test_handle_killed_worker(container, logger):
    if False:
        i = 10
        return i + 15
    dep = get_extension(container, Entrypoint)
    container.spawn_worker(dep, ['sleep'], {})
    assert len(container._worker_threads) == 1
    (worker_gt,) = container._worker_threads.values()
    worker_gt.kill()
    assert logger.debug.call_args == call('%s thread killed by container', container)
    assert not container._died.ready()

def test_spawned_thread_kills_container(container):
    if False:
        while True:
            i = 10

    def raise_error():
        if False:
            for i in range(10):
                print('nop')
        raise Exception('foobar')
    container.start()
    container.spawn_managed_thread(raise_error)
    with pytest.raises(Exception) as exc_info:
        container.wait()
    assert exc_info.value.args == ('foobar',)

def test_spawned_thread_causes_container_to_kill_other_thread(container):
    if False:
        print('Hello World!')
    killed_by_error_raised = Event()

    def raise_error():
        if False:
            return 10
        raise Exception('foobar')

    def wait_forever():
        if False:
            for i in range(10):
                print('nop')
        try:
            Event().wait()
        except:
            killed_by_error_raised.send()
            raise
    container.start()
    container.spawn_managed_thread(wait_forever)
    container.spawn_managed_thread(raise_error)
    with Timeout(1):
        killed_by_error_raised.wait()

def test_container_only_killed_once(container):
    if False:
        for i in range(10):
            print('nop')

    class Broken(Exception):
        pass
    exc = Broken('foobar')

    def raise_error():
        if False:
            return 10
        raise exc
    with patch.object(container, '_kill_managed_threads', autospec=True) as kill_managed_threads:
        with patch.object(container, 'kill', wraps=container.kill) as kill:
            container.start()
            container.spawn_managed_thread(raise_error)
            container.spawn_managed_thread(raise_error)
            with pytest.raises(Broken):
                container.wait()
            assert kill.call_args_list == [call((Broken, exc, ANY)), call((Broken, exc, ANY))]
            assert kill_managed_threads.call_count == 1

def test_container_stop_kills_remaining_managed_threads(container, logger):
    if False:
        i = 10
        return i + 15
    ' Verify any remaining managed threads are killed when a container\n    is stopped.\n    '

    def sleep_forever():
        if False:
            for i in range(10):
                print('nop')
        while True:
            sleep()
    container.start()
    container.spawn_managed_thread(sleep_forever)
    container.spawn_managed_thread(sleep_forever)
    container.stop()
    assert logger.warning.call_args_list == [call('killing %s managed thread(s)', 2), call('killing managed thread `%s`', 'sleep_forever'), call('killing managed thread `%s`', 'sleep_forever')]
    assert logger.debug.call_args_list == [call('starting %s', container), call('stopping %s', container), call('%s thread killed by container', container), call('%s thread killed by container', container)]

def test_container_kill_kills_remaining_managed_threads(container, logger):
    if False:
        for i in range(10):
            print('nop')
    ' Verify any remaining managed threads are killed when a container\n    is killed.\n    '

    def sleep_forever():
        if False:
            i = 10
            return i + 15
        while True:
            sleep()
    container.start()
    container.spawn_managed_thread(sleep_forever)
    container.spawn_managed_thread(sleep_forever)
    container.kill()
    assert logger.warning.call_args_list == [call('killing %s managed thread(s)', 2), call('killing managed thread `%s`', 'sleep_forever'), call('killing managed thread `%s`', 'sleep_forever')]
    assert logger.info.call_args_list == [call('killing %s', container)]
    assert logger.debug.call_args_list == [call('starting %s', container), call('%s thread killed by container', container), call('%s thread killed by container', container)]

def test_kill_bad_dependency(container):
    if False:
        i = 10
        return i + 15
    " Verify that an exception from a badly-behaved dependency.kill()\n    doesn't stop the container's kill process.\n    "
    dep = get_extension(container, DependencyProvider)
    with patch.object(dep, 'kill') as dep_kill:
        dep_kill.side_effect = Exception('dependency error')
        container.start()
        try:
            raise Exception('container error')
        except:
            exc_info = sys.exc_info()
        container.kill(exc_info)
        with pytest.raises(Exception) as exc_info:
            container.wait()
        assert str(exc_info.value) == 'container error'

def test_stop_during_kill(container, logger):
    if False:
        for i in range(10):
            print('nop')
    ' Verify we handle the race condition when a runner tries to stop\n    a container while it is being killed.\n    '
    with patch.object(container, '_kill_managed_threads', autospec=True) as kill_managed_threads:
        kill_managed_threads.side_effect = eventlet.sleep
        try:
            raise Exception('error')
        except:
            exc_info = sys.exc_info()
        eventlet.spawn(container.kill, exc_info)
        eventlet.spawn(container.stop)
        with pytest.raises(Exception):
            container.wait()
        assert logger.debug.call_args_list == [call('already being killed %s', container)]

def test_get_service_name():
    if False:
        while True:
            i = 10

    class Service:
        name = 'str'

    class UnicodeService:
        name = u'unicøde'

    class BadNameService:
        name = object()

    class AnonymousService:
        pass
    assert get_service_name(Service) == 'str'
    assert get_service_name(UnicodeService) == u'unicøde'
    with pytest.raises(ConfigurationError) as exc_info:
        get_service_name(BadNameService)
    assert str(exc_info.value) == 'Service name attribute must be a string (test.test_container.BadNameService.name)'
    with pytest.raises(ConfigurationError) as exc_info:
        get_service_name(AnonymousService)
    assert str(exc_info.value) == 'Service class must define a `name` attribute (test.test_container.AnonymousService)'

def test_logging_managed_threads(container, logger):
    if False:
        return 10

    def wait():
        if False:
            print('Hello World!')
        Event().wait()
    container.spawn_managed_thread(wait)
    container.spawn_managed_thread(partial(wait))
    container.spawn_managed_thread(wait, identifier='named')
    container.stop()
    call_args_list = logger.warning.call_args_list
    assert call('killing %s managed thread(s)', 3) in call_args_list
    assert call('killing managed thread `%s`', 'wait') in call_args_list
    assert call('killing managed thread `%s`', '<unknown>') in call_args_list
    assert call('killing managed thread `%s`', 'named') in call_args_list