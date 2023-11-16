import asyncio
import logging
from time import sleep
import pytest
from sanic import Sanic
from sanic.exceptions import ServiceUnavailable
from sanic.log import LOGGING_CONFIG_DEFAULTS
from sanic.response import text

@pytest.fixture
def response_timeout_app():
    if False:
        while True:
            i = 10
    app = Sanic('test_response_timeout')
    app.config.RESPONSE_TIMEOUT = 1

    @app.route('/1')
    async def handler_1(request):
        await asyncio.sleep(2)
        return text('OK')

    @app.exception(ServiceUnavailable)
    def handler_exception(request, exception):
        if False:
            for i in range(10):
                print('nop')
        return text('Response Timeout from error_handler.', 503)
    return app

@pytest.fixture
def response_timeout_default_app():
    if False:
        print('Hello World!')
    app = Sanic('test_response_timeout_default')
    app.config.RESPONSE_TIMEOUT = 1

    @app.route('/1')
    async def handler_2(request):
        await asyncio.sleep(2)
        return text('OK')
    return app

@pytest.fixture
def response_handler_cancelled_app():
    if False:
        while True:
            i = 10
    app = Sanic('test_response_handler_cancelled')
    app.config.RESPONSE_TIMEOUT = 1
    app.ctx.flag = False

    @app.exception(asyncio.CancelledError)
    def handler_cancelled(request, exception):
        if False:
            print('Hello World!')
        response_handler_cancelled_app.ctx.flag = True
        return text('App received CancelledError!', 500)

    @app.route('/1')
    async def handler_3(request):
        await asyncio.sleep(2)
        return text('OK')
    return app

def test_server_error_response_timeout(response_timeout_app):
    if False:
        for i in range(10):
            print('nop')
    (request, response) = response_timeout_app.test_client.get('/1')
    assert response.status == 503
    assert response.text == 'Response Timeout from error_handler.'

def test_default_server_error_response_timeout(response_timeout_default_app):
    if False:
        return 10
    (request, response) = response_timeout_default_app.test_client.get('/1')
    assert response.status == 503
    assert 'Response Timeout' in response.text

def test_response_handler_cancelled(response_handler_cancelled_app):
    if False:
        for i in range(10):
            print('nop')
    (request, response) = response_handler_cancelled_app.test_client.get('/1')
    assert response.status == 503
    assert 'Response Timeout' in response.text
    assert response_handler_cancelled_app.ctx.flag is False

def test_response_timeout_not_applied(caplog):
    if False:
        for i in range(10):
            print('nop')
    modified_config = LOGGING_CONFIG_DEFAULTS
    modified_config['loggers']['sanic.root']['level'] = 'DEBUG'
    app = Sanic('test_logging', log_config=modified_config)
    app.config.RESPONSE_TIMEOUT = 1
    app.ctx.event = asyncio.Event()

    @app.websocket('/ws')
    async def ws_handler(request, ws):
        sleep(2)
        await asyncio.sleep(0)
        request.app.ctx.event.set()
    with caplog.at_level(logging.DEBUG):
        _ = app.test_client.websocket('/ws')
    assert app.ctx.event.is_set()
    assert ('sanic.root', 10, 'Handling websocket. Timeouts disabled.') in caplog.record_tuples