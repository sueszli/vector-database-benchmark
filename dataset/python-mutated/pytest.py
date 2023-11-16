"""Fixtures and testing utilities for :pypi:`pytest <pytest>`."""
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union
import pytest
if TYPE_CHECKING:
    from celery import Celery
    from ..worker import WorkController
else:
    Celery = WorkController = object
NO_WORKER = os.environ.get('NO_WORKER')

def pytest_configure(config):
    if False:
        print('Hello World!')
    'Register additional pytest configuration.'
    config.addinivalue_line('markers', 'celery(**overrides): override celery configuration for a test case')

@contextmanager
def _create_app(enable_logging=False, use_trap=False, parameters=None, **config):
    if False:
        print('Hello World!')
    'Utility context used to setup Celery app for pytest fixtures.'
    from .testing.app import TestApp, setup_default_app
    parameters = {} if not parameters else parameters
    test_app = TestApp(set_as_current=False, enable_logging=enable_logging, config=config, **parameters)
    with setup_default_app(test_app, use_trap=use_trap):
        yield test_app

@pytest.fixture(scope='session')
def use_celery_app_trap():
    if False:
        for i in range(10):
            print('nop')
    'You can override this fixture to enable the app trap.\n\n    The app trap raises an exception whenever something attempts\n    to use the current or default apps.\n    '
    return False

@pytest.fixture(scope='session')
def celery_session_app(request, celery_config, celery_parameters, celery_enable_logging, use_celery_app_trap):
    if False:
        i = 10
        return i + 15
    'Session Fixture: Return app for session fixtures.'
    mark = request.node.get_closest_marker('celery')
    config = dict(celery_config, **mark.kwargs if mark else {})
    with _create_app(enable_logging=celery_enable_logging, use_trap=use_celery_app_trap, parameters=celery_parameters, **config) as app:
        if not use_celery_app_trap:
            app.set_default()
            app.set_current()
        yield app

@pytest.fixture(scope='session')
def celery_session_worker(request, celery_session_app, celery_includes, celery_class_tasks, celery_worker_pool, celery_worker_parameters):
    if False:
        while True:
            i = 10
    'Session Fixture: Start worker that lives throughout test suite.'
    from .testing import worker
    if not NO_WORKER:
        for module in celery_includes:
            celery_session_app.loader.import_task_module(module)
        for class_task in celery_class_tasks:
            celery_session_app.register_task(class_task)
        with worker.start_worker(celery_session_app, pool=celery_worker_pool, **celery_worker_parameters) as w:
            yield w

@pytest.fixture(scope='session')
def celery_enable_logging():
    if False:
        return 10
    'You can override this fixture to enable logging.'
    return False

@pytest.fixture(scope='session')
def celery_includes():
    if False:
        while True:
            i = 10
    'You can override this include modules when a worker start.\n\n    You can have this return a list of module names to import,\n    these can be task modules, modules registering signals, and so on.\n    '
    return ()

@pytest.fixture(scope='session')
def celery_worker_pool():
    if False:
        i = 10
        return i + 15
    'You can override this fixture to set the worker pool.\n\n    The "solo" pool is used by default, but you can set this to\n    return e.g. "prefork".\n    '
    return 'solo'

@pytest.fixture(scope='session')
def celery_config():
    if False:
        print('Hello World!')
    'Redefine this fixture to configure the test Celery app.\n\n    The config returned by your fixture will then be used\n    to configure the :func:`celery_app` fixture.\n    '
    return {}

@pytest.fixture(scope='session')
def celery_parameters():
    if False:
        while True:
            i = 10
    'Redefine this fixture to change the init parameters of test Celery app.\n\n    The dict returned by your fixture will then be used\n    as parameters when instantiating :class:`~celery.Celery`.\n    '
    return {}

@pytest.fixture(scope='session')
def celery_worker_parameters():
    if False:
        i = 10
        return i + 15
    'Redefine this fixture to change the init parameters of Celery workers.\n\n    This can be used e. g. to define queues the worker will consume tasks from.\n\n    The dict returned by your fixture will then be used\n    as parameters when instantiating :class:`~celery.worker.WorkController`.\n    '
    return {}

@pytest.fixture()
def celery_app(request, celery_config, celery_parameters, celery_enable_logging, use_celery_app_trap):
    if False:
        i = 10
        return i + 15
    'Fixture creating a Celery application instance.'
    mark = request.node.get_closest_marker('celery')
    config = dict(celery_config, **mark.kwargs if mark else {})
    with _create_app(enable_logging=celery_enable_logging, use_trap=use_celery_app_trap, parameters=celery_parameters, **config) as app:
        yield app

@pytest.fixture(scope='session')
def celery_class_tasks():
    if False:
        return 10
    'Redefine this fixture to register tasks with the test Celery app.'
    return []

@pytest.fixture()
def celery_worker(request, celery_app, celery_includes, celery_worker_pool, celery_worker_parameters):
    if False:
        return 10
    'Fixture: Start worker in a thread, stop it when the test returns.'
    from .testing import worker
    if not NO_WORKER:
        for module in celery_includes:
            celery_app.loader.import_task_module(module)
        with worker.start_worker(celery_app, pool=celery_worker_pool, **celery_worker_parameters) as w:
            yield w

@pytest.fixture()
def depends_on_current_app(celery_app):
    if False:
        while True:
            i = 10
    'Fixture that sets app as current.'
    celery_app.set_current()