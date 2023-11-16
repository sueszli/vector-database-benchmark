import logging
from os import environ
from unittest.mock import Mock, patch
import pytest
from sanic.app import Sanic
from sanic.worker.loader import AppLoader
from sanic.worker.multiplexer import WorkerMultiplexer
from sanic.worker.process import Worker, WorkerProcess
from sanic.worker.serve import worker_serve

@pytest.fixture
def mock_app():
    if False:
        i = 10
        return i + 15
    app = Mock()
    server_info = Mock()
    server_info.settings = {'app': app}
    app.state.workers = 1
    app.listeners = {'main_process_ready': []}
    app.get_motd_data.return_value = ({'packages': ''}, {})
    app.state.server_info = [server_info]
    return app

def args(app, **kwargs):
    if False:
        while True:
            i = 10
    params = {**kwargs}
    params.setdefault('host', '127.0.0.1')
    params.setdefault('port', 9999)
    params.setdefault('app_name', 'test_config_app')
    params.setdefault('monitor_publisher', None)
    params.setdefault('app_loader', AppLoader(factory=lambda : app))
    return params

def test_config_app(mock_app: Mock):
    if False:
        while True:
            i = 10
    with patch('sanic.worker.serve._serve_http_1'):
        worker_serve(**args(mock_app, config={'FOO': 'BAR'}))
    mock_app.update_config.assert_called_once_with({'FOO': 'BAR'})

def test_bad_process(mock_app: Mock, caplog):
    if False:
        while True:
            i = 10
    environ['SANIC_WORKER_NAME'] = Worker.WORKER_PREFIX + WorkerProcess.SERVER_LABEL + '-FOO'
    message = 'No restart publisher found in worker process'
    with pytest.raises(RuntimeError, match=message):
        worker_serve(**args(mock_app))
    message = 'No worker state found in worker process'
    publisher = Mock()
    with caplog.at_level(logging.ERROR):
        worker_serve(**args(mock_app, monitor_publisher=publisher))
    assert ('sanic.error', logging.ERROR, message) in caplog.record_tuples
    publisher.send.assert_called_once_with('__TERMINATE_EARLY__')
    del environ['SANIC_WORKER_NAME']

def test_has_multiplexer(app: Sanic):
    if False:
        for i in range(10):
            print('nop')
    environ['SANIC_WORKER_NAME'] = Worker.WORKER_PREFIX + WorkerProcess.SERVER_LABEL + '-FOO'
    Sanic.register_app(app)
    with patch('sanic.worker.serve._serve_http_1'):
        worker_serve(**args(app, monitor_publisher=Mock(), worker_state=Mock()))
    assert isinstance(app.multiplexer, WorkerMultiplexer)
    del environ['SANIC_WORKER_NAME']

@patch('sanic.mixins.startup.WorkerManager')
def test_serve_app_implicit(wm: Mock, app):
    if False:
        i = 10
        return i + 15
    app.prepare()
    Sanic.serve()
    wm.call_args[0] == app.state.workers

@patch('sanic.mixins.startup.WorkerManager')
def test_serve_app_explicit(wm: Mock, mock_app):
    if False:
        for i in range(10):
            print('nop')
    Sanic.serve(mock_app)
    wm.call_args[0] == mock_app.state.workers

@patch('sanic.mixins.startup.WorkerManager')
def test_serve_app_loader(wm: Mock, mock_app):
    if False:
        return 10
    Sanic.serve(app_loader=AppLoader(factory=lambda : mock_app))
    wm.call_args[0] == mock_app.state.workers

@patch('sanic.mixins.startup.WorkerManager')
def test_serve_app_factory(wm: Mock, mock_app):
    if False:
        while True:
            i = 10
    Sanic.serve(factory=lambda : mock_app)
    wm.call_args[0] == mock_app.state.workers

@patch('sanic.mixins.startup.WorkerManager')
@pytest.mark.parametrize('config', (True, False))
def test_serve_with_inspector(WorkerManager: Mock, mock_app: Mock, config: bool):
    if False:
        print('Hello World!')
    Inspector = Mock()
    mock_app.config.INSPECTOR = config
    mock_app.inspector_class = Inspector
    inspector = Mock()
    Inspector.return_value = inspector
    WorkerManager.return_value = WorkerManager
    Sanic.serve(mock_app)
    if config:
        Inspector.assert_called_once()
        WorkerManager.manage.assert_called_once_with('Inspector', inspector, {}, transient=False)
    else:
        Inspector.assert_not_called()
        WorkerManager.manage.assert_not_called()