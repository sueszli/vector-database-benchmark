import functools
import pytest
from fastapi import APIRouter, Depends, FastAPI, Header, WebSocket, WebSocketDisconnect, status
from fastapi.middleware import Middleware
from fastapi.testclient import TestClient
router = APIRouter()
prefix_router = APIRouter()
native_prefix_route = APIRouter(prefix='/native')
app = FastAPI()

@app.websocket_route('/')
async def index(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text('Hello, world!')
    await websocket.close()

@router.websocket_route('/router')
async def routerindex(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text('Hello, router!')
    await websocket.close()

@prefix_router.websocket_route('/')
async def routerprefixindex(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text('Hello, router with prefix!')
    await websocket.close()

@router.websocket('/router2')
async def routerindex2(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text('Hello, router!')
    await websocket.close()

@router.websocket('/router/{pathparam:path}')
async def routerindexparams(websocket: WebSocket, pathparam: str, queryparam: str):
    await websocket.accept()
    await websocket.send_text(pathparam)
    await websocket.send_text(queryparam)
    await websocket.close()

async def ws_dependency():
    return 'Socket Dependency'

@router.websocket('/router-ws-depends/')
async def router_ws_decorator_depends(websocket: WebSocket, data=Depends(ws_dependency)):
    await websocket.accept()
    await websocket.send_text(data)
    await websocket.close()

@native_prefix_route.websocket('/')
async def router_native_prefix_ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text('Hello, router with native prefix!')
    await websocket.close()

async def ws_dependency_err():
    raise NotImplementedError()

@router.websocket('/depends-err/')
async def router_ws_depends_err(websocket: WebSocket, data=Depends(ws_dependency_err)):
    pass

async def ws_dependency_validate(x_missing: str=Header()):
    pass

@router.websocket('/depends-validate/')
async def router_ws_depends_validate(websocket: WebSocket, data=Depends(ws_dependency_validate)):
    pass

class CustomError(Exception):
    pass

@router.websocket('/custom_error/')
async def router_ws_custom_error(websocket: WebSocket):
    raise CustomError()

def make_app(app=None, **kwargs):
    if False:
        print('Hello World!')
    app = app or FastAPI(**kwargs)
    app.include_router(router)
    app.include_router(prefix_router, prefix='/prefix')
    app.include_router(native_prefix_route)
    return app
app = make_app(app)

def test_app():
    if False:
        while True:
            i = 10
    client = TestClient(app)
    with client.websocket_connect('/') as websocket:
        data = websocket.receive_text()
        assert data == 'Hello, world!'

def test_router():
    if False:
        return 10
    client = TestClient(app)
    with client.websocket_connect('/router') as websocket:
        data = websocket.receive_text()
        assert data == 'Hello, router!'

def test_prefix_router():
    if False:
        i = 10
        return i + 15
    client = TestClient(app)
    with client.websocket_connect('/prefix/') as websocket:
        data = websocket.receive_text()
        assert data == 'Hello, router with prefix!'

def test_native_prefix_router():
    if False:
        while True:
            i = 10
    client = TestClient(app)
    with client.websocket_connect('/native/') as websocket:
        data = websocket.receive_text()
        assert data == 'Hello, router with native prefix!'

def test_router2():
    if False:
        while True:
            i = 10
    client = TestClient(app)
    with client.websocket_connect('/router2') as websocket:
        data = websocket.receive_text()
        assert data == 'Hello, router!'

def test_router_ws_depends():
    if False:
        return 10
    client = TestClient(app)
    with client.websocket_connect('/router-ws-depends/') as websocket:
        assert websocket.receive_text() == 'Socket Dependency'

def test_router_ws_depends_with_override():
    if False:
        for i in range(10):
            print('nop')
    client = TestClient(app)
    app.dependency_overrides[ws_dependency] = lambda : 'Override'
    with client.websocket_connect('/router-ws-depends/') as websocket:
        assert websocket.receive_text() == 'Override'

def test_router_with_params():
    if False:
        print('Hello World!')
    client = TestClient(app)
    with client.websocket_connect('/router/path/to/file?queryparam=a_query_param') as websocket:
        data = websocket.receive_text()
        assert data == 'path/to/file'
        data = websocket.receive_text()
        assert data == 'a_query_param'

def test_wrong_uri():
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify that a websocket connection to a non-existent endpoing returns in a shutdown\n    '
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect('/no-router/'):
            pass
    assert e.value.code == status.WS_1000_NORMAL_CLOSURE

def websocket_middleware(middleware_func):
    if False:
        while True:
            i = 10
    '\n    Helper to create a Starlette pure websocket middleware\n    '

    def middleware_constructor(app):
        if False:
            print('Hello World!')

        @functools.wraps(app)
        async def wrapped_app(scope, receive, send):
            if scope['type'] != 'websocket':
                return await app(scope, receive, send)

            async def call_next():
                return await app(scope, receive, send)
            websocket = WebSocket(scope, receive=receive, send=send)
            return await middleware_func(websocket, call_next)
        return wrapped_app
    return middleware_constructor

def test_depend_validation():
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify that a validation in a dependency invokes the correct exception handler\n    '
    caught = []

    @websocket_middleware
    async def catcher(websocket, call_next):
        try:
            return await call_next()
        except Exception as e:
            caught.append(e)
            raise
    myapp = make_app(middleware=[Middleware(catcher)])
    client = TestClient(myapp)
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect('/depends-validate/'):
            pass
    assert e.value.code == status.WS_1008_POLICY_VIOLATION
    assert caught == []

def test_depend_err_middleware():
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify that it is possible to write custom WebSocket middleware to catch errors\n    '

    @websocket_middleware
    async def errorhandler(websocket: WebSocket, call_next):
        try:
            return await call_next()
        except Exception as e:
            await websocket.close(code=status.WS_1006_ABNORMAL_CLOSURE, reason=repr(e))
    myapp = make_app(middleware=[Middleware(errorhandler)])
    client = TestClient(myapp)
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect('/depends-err/'):
            pass
    assert e.value.code == status.WS_1006_ABNORMAL_CLOSURE
    assert 'NotImplementedError' in e.value.reason

def test_depend_err_handler():
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify that it is possible to write custom WebSocket middleware to catch errors\n    '

    async def custom_handler(websocket: WebSocket, exc: CustomError) -> None:
        await websocket.close(1002, 'foo')
    myapp = make_app(exception_handlers={CustomError: custom_handler})
    client = TestClient(myapp)
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect('/custom_error/'):
            pass
    assert e.value.code == 1002
    assert 'foo' in e.value.reason