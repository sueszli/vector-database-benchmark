import sys
from multiprocessing import Event
from os import environ, getpid
from typing import Any, Dict, Type, Union
from unittest.mock import Mock
import pytest
from sanic import Sanic
from sanic.compat import use_context
from sanic.worker.multiplexer import WorkerMultiplexer
from sanic.worker.state import WorkerState

@pytest.fixture
def monitor_publisher():
    if False:
        return 10
    return Mock()

@pytest.fixture
def worker_state():
    if False:
        for i in range(10):
            print('nop')
    return {}

@pytest.fixture
def m(monitor_publisher, worker_state):
    if False:
        for i in range(10):
            print('nop')
    environ['SANIC_WORKER_NAME'] = 'Test'
    worker_state['Test'] = {}
    yield WorkerMultiplexer(monitor_publisher, worker_state)
    del environ['SANIC_WORKER_NAME']

@pytest.mark.skipif(sys.platform not in ('linux', 'darwin'), reason='This test requires fork context')
def test_has_multiplexer_default(app: Sanic):
    if False:
        return 10
    event = Event()

    @app.main_process_start
    async def setup(app, _):
        app.shared_ctx.event = event

    @app.after_server_start
    def stop(app):
        if False:
            print('Hello World!')
        if hasattr(app, 'm') and isinstance(app.m, WorkerMultiplexer):
            app.shared_ctx.event.set()
        app.stop()
    with use_context('fork'):
        app.run()
    assert event.is_set()

def test_not_have_multiplexer_single(app: Sanic):
    if False:
        print('Hello World!')
    event = Event()

    @app.main_process_start
    async def setup(app, _):
        app.shared_ctx.event = event

    @app.after_server_start
    def stop(app):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(app, 'm') and isinstance(app.m, WorkerMultiplexer):
            app.shared_ctx.event.set()
        app.stop()
    app.run(single_process=True)
    assert not event.is_set()

def test_ack(worker_state: Dict[str, Any], m: WorkerMultiplexer):
    if False:
        for i in range(10):
            print('nop')
    worker_state['Test'] = {'foo': 'bar'}
    m.ack()
    assert worker_state['Test'] == {'foo': 'bar', 'state': 'ACKED'}

def test_restart_self(monitor_publisher: Mock, m: WorkerMultiplexer):
    if False:
        for i in range(10):
            print('nop')
    m.restart()
    monitor_publisher.send.assert_called_once_with('Test:')

def test_restart_foo(monitor_publisher: Mock, m: WorkerMultiplexer):
    if False:
        print('Hello World!')
    m.restart('foo')
    monitor_publisher.send.assert_called_once_with('foo:')

def test_reload_alias(monitor_publisher: Mock, m: WorkerMultiplexer):
    if False:
        print('Hello World!')
    m.reload()
    monitor_publisher.send.assert_called_once_with('Test:')

def test_terminate(monitor_publisher: Mock, m: WorkerMultiplexer):
    if False:
        for i in range(10):
            print('nop')
    m.terminate()
    monitor_publisher.send.assert_called_once_with('__TERMINATE__')

def test_scale(monitor_publisher: Mock, m: WorkerMultiplexer):
    if False:
        while True:
            i = 10
    m.scale(99)
    monitor_publisher.send.assert_called_once_with('__SCALE__:99')

def test_properties(monitor_publisher: Mock, worker_state: Dict[str, Any], m: WorkerMultiplexer):
    if False:
        i = 10
        return i + 15
    assert m.reload == m.restart
    assert m.pid == getpid()
    assert m.name == 'Test'
    assert m.workers == worker_state
    assert m.state == worker_state['Test']
    assert isinstance(m.state, WorkerState)

@pytest.mark.parametrize('params,expected', (({}, 'Test:'), ({'name': 'foo'}, 'foo:'), ({'all_workers': True}, '__ALL_PROCESSES__:'), ({'zero_downtime': True}, 'Test::STARTUP_FIRST'), ({'name': 'foo', 'all_workers': True}, ValueError), ({'name': 'foo', 'zero_downtime': True}, 'foo::STARTUP_FIRST'), ({'all_workers': True, 'zero_downtime': True}, '__ALL_PROCESSES__::STARTUP_FIRST'), ({'name': 'foo', 'all_workers': True, 'zero_downtime': True}, ValueError)))
def test_restart_params(monitor_publisher: Mock, m: WorkerMultiplexer, params: Dict[str, Any], expected: Union[str, Type[Exception]]):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(expected, str):
        m.restart(**params)
        monitor_publisher.send.assert_called_once_with(expected)
    else:
        with pytest.raises(expected):
            m.restart(**params)