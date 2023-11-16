import asyncio
import importlib
import pytest
from uvicorn.config import Config
from uvicorn.loops.auto import auto_loop_setup
from uvicorn.main import ServerState
from uvicorn.protocols.http.auto import AutoHTTPProtocol
from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol
try:
    importlib.import_module('uvloop')
    expected_loop = 'uvloop'
except ImportError:
    expected_loop = 'asyncio'
try:
    importlib.import_module('httptools')
    expected_http = 'HttpToolsProtocol'
except ImportError:
    expected_http = 'H11Protocol'
try:
    importlib.import_module('websockets')
    expected_websockets = 'WebSocketProtocol'
except ImportError:
    expected_websockets = 'WSProtocol'

async def app(scope, receive, send):
    pass

def test_loop_auto():
    if False:
        while True:
            i = 10
    auto_loop_setup()
    policy = asyncio.get_event_loop_policy()
    assert isinstance(policy, asyncio.events.BaseDefaultEventLoopPolicy)
    assert type(policy).__module__.startswith(expected_loop)

@pytest.mark.anyio
async def test_http_auto():
    config = Config(app=app)
    server_state = ServerState()
    protocol = AutoHTTPProtocol(config=config, server_state=server_state, app_state={})
    assert type(protocol).__name__ == expected_http

@pytest.mark.anyio
async def test_websocket_auto():
    config = Config(app=app)
    server_state = ServerState()
    assert AutoWebSocketsProtocol is not None
    protocol = AutoWebSocketsProtocol(config=config, server_state=server_state, app_state={})
    assert type(protocol).__name__ == expected_websockets