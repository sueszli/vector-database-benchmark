import itertools
import sys
from asyncio import current_task as asyncio_current_task
from contextlib import asynccontextmanager
from typing import Callable
import anyio
import pytest
import sniffio
import trio.lowlevel
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

def mock_service_endpoint(request):
    if False:
        print('Hello World!')
    return JSONResponse({'mock': 'example'})
mock_service = Starlette(routes=[Route('/', endpoint=mock_service_endpoint)])

def current_task():
    if False:
        return 10
    asynclib_name = sniffio.current_async_library()
    if asynclib_name == 'trio':
        return trio.lowlevel.current_task()
    if asynclib_name == 'asyncio':
        task = asyncio_current_task()
        if task is None:
            raise RuntimeError('must be called from a running task')
        return task
    raise RuntimeError(f'unsupported asynclib={asynclib_name}')

def startup():
    if False:
        print('Hello World!')
    raise RuntimeError()

def test_use_testclient_in_endpoint(test_client_factory):
    if False:
        print('Hello World!')
    '\n    We should be able to use the test client within applications.\n\n    This is useful if we need to mock out other services,\n    during tests or in development.\n    '

    def homepage(request):
        if False:
            i = 10
            return i + 15
        client = test_client_factory(mock_service)
        response = client.get('/')
        return JSONResponse(response.json())
    app = Starlette(routes=[Route('/', endpoint=homepage)])
    client = test_client_factory(app)
    response = client.get('/')
    assert response.json() == {'mock': 'example'}

def test_testclient_headers_behavior():
    if False:
        return 10
    '\n    We should be able to use the test client with user defined headers.\n\n    This is useful if we need to set custom headers for authentication\n    during tests or in development.\n    '
    client = TestClient(mock_service)
    assert client.headers.get('user-agent') == 'testclient'
    client = TestClient(mock_service, headers={'user-agent': 'non-default-agent'})
    assert client.headers.get('user-agent') == 'non-default-agent'
    client = TestClient(mock_service, headers={'Authentication': 'Bearer 123'})
    assert client.headers.get('user-agent') == 'testclient'
    assert client.headers.get('Authentication') == 'Bearer 123'

def test_use_testclient_as_contextmanager(test_client_factory, anyio_backend_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    This test asserts a number of properties that are important for an\n    app level task_group\n    '
    counter = itertools.count()
    identity_runvar = anyio.lowlevel.RunVar[int]('identity_runvar')

    def get_identity():
        if False:
            return 10
        try:
            return identity_runvar.get()
        except LookupError:
            token = next(counter)
            identity_runvar.set(token)
            return token
    startup_task = object()
    startup_loop = None
    shutdown_task = object()
    shutdown_loop = None

    @asynccontextmanager
    async def lifespan_context(app):
        nonlocal startup_task, startup_loop, shutdown_task, shutdown_loop
        startup_task = current_task()
        startup_loop = get_identity()
        async with anyio.create_task_group() as app.task_group:
            yield
        shutdown_task = current_task()
        shutdown_loop = get_identity()

    async def loop_id(request):
        return JSONResponse(get_identity())
    app = Starlette(lifespan=lifespan_context, routes=[Route('/loop_id', endpoint=loop_id)])
    client = test_client_factory(app)
    with client:
        assert client.get('/loop_id').json() == 0
        assert client.get('/loop_id').json() == 0
    assert startup_loop == 0
    assert shutdown_loop == 0
    assert startup_task is shutdown_task
    assert client.get('/loop_id').json() == 1
    assert client.get('/loop_id').json() == 2
    first_task = startup_task
    with client:
        assert client.get('/loop_id').json() == 3
        assert client.get('/loop_id').json() == 3
    assert startup_loop == 3
    assert shutdown_loop == 3
    assert startup_task is shutdown_task
    assert first_task is not startup_task

def test_error_on_startup(test_client_factory):
    if False:
        for i in range(10):
            print('nop')
    with pytest.deprecated_call(match='The on_startup and on_shutdown parameters are deprecated'):
        startup_error_app = Starlette(on_startup=[startup])
    with pytest.raises(RuntimeError):
        with test_client_factory(startup_error_app):
            pass

def test_exception_in_middleware(test_client_factory):
    if False:
        return 10

    class MiddlewareException(Exception):
        pass

    class BrokenMiddleware:

        def __init__(self, app):
            if False:
                i = 10
                return i + 15
            self.app = app

        async def __call__(self, scope, receive, send):
            raise MiddlewareException()
    broken_middleware = Starlette(middleware=[Middleware(BrokenMiddleware)])
    with pytest.raises(MiddlewareException):
        with test_client_factory(broken_middleware):
            pass

def test_testclient_asgi2(test_client_factory):
    if False:
        i = 10
        return i + 15

    def app(scope):
        if False:
            i = 10
            return i + 15

        async def inner(receive, send):
            await send({'type': 'http.response.start', 'status': 200, 'headers': [[b'content-type', b'text/plain']]})
            await send({'type': 'http.response.body', 'body': b'Hello, world!'})
        return inner
    client = test_client_factory(app)
    response = client.get('/')
    assert response.text == 'Hello, world!'

def test_testclient_asgi3(test_client_factory):
    if False:
        for i in range(10):
            print('nop')

    async def app(scope, receive, send):
        await send({'type': 'http.response.start', 'status': 200, 'headers': [[b'content-type', b'text/plain']]})
        await send({'type': 'http.response.body', 'body': b'Hello, world!'})
    client = test_client_factory(app)
    response = client.get('/')
    assert response.text == 'Hello, world!'

def test_websocket_blocking_receive(test_client_factory):
    if False:
        while True:
            i = 10

    def app(scope):
        if False:
            print('Hello World!')

        async def respond(websocket):
            await websocket.send_json({'message': 'test'})

        async def asgi(receive, send):
            websocket = WebSocket(scope, receive=receive, send=send)
            await websocket.accept()
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(respond, websocket)
                try:
                    await websocket.receive_json()
                except WebSocketDisconnect:
                    pass
        return asgi
    client = test_client_factory(app)
    with client.websocket_connect('/') as websocket:
        data = websocket.receive_json()
        assert data == {'message': 'test'}

def test_client(test_client_factory):
    if False:
        return 10

    async def app(scope, receive, send):
        client = scope.get('client')
        assert client is not None
        (host, port) = client
        response = JSONResponse({'host': host, 'port': port})
        await response(scope, receive, send)
    client = test_client_factory(app)
    response = client.get('/')
    assert response.json() == {'host': 'testclient', 'port': 50000}

@pytest.mark.parametrize('param', ('2020-07-14T00:00:00+00:00', 'España', 'voilà'))
def test_query_params(test_client_factory, param: str):
    if False:
        while True:
            i = 10

    def homepage(request):
        if False:
            i = 10
            return i + 15
        return Response(request.query_params['param'])
    app = Starlette(routes=[Route('/', endpoint=homepage)])
    client = test_client_factory(app)
    response = client.get('/', params={'param': param})
    assert response.text == param

@pytest.mark.parametrize('domain, ok', [pytest.param('testserver', True, marks=[pytest.mark.xfail(sys.version_info < (3, 11), reason='Fails due to domain handling in http.cookiejar module (see #2152)')]), ('testserver.local', True), ('localhost', False), ('example.com', False)])
def test_domain_restricted_cookies(test_client_factory, domain, ok):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that test client discards domain restricted cookies which do not match the\n    base_url of the testclient (`http://testserver` by default).\n\n    The domain `testserver.local` works because the Python http.cookiejar module derives\n    the "effective domain" by appending `.local` to non-dotted request domains\n    in accordance with RFC 2965.\n    '

    async def app(scope, receive, send):
        response = Response('Hello, world!', media_type='text/plain')
        response.set_cookie('mycookie', 'myvalue', path='/', domain=domain)
        await response(scope, receive, send)
    client = test_client_factory(app)
    response = client.get('/')
    cookie_set = len(response.cookies) == 1
    assert cookie_set == ok

def test_forward_follow_redirects(test_client_factory):
    if False:
        for i in range(10):
            print('nop')

    async def app(scope, receive, send):
        if '/ok' in scope['path']:
            response = Response('ok')
        else:
            response = RedirectResponse('/ok')
        await response(scope, receive, send)
    client = test_client_factory(app, follow_redirects=True)
    response = client.get('/')
    assert response.status_code == 200

def test_forward_nofollow_redirects(test_client_factory):
    if False:
        return 10

    async def app(scope, receive, send):
        response = RedirectResponse('/ok')
        await response(scope, receive, send)
    client = test_client_factory(app, follow_redirects=False)
    response = client.get('/')
    assert response.status_code == 307

def test_with_duplicate_headers(test_client_factory: Callable[[Starlette], TestClient]):
    if False:
        for i in range(10):
            print('nop')

    def homepage(request: Request) -> JSONResponse:
        if False:
            i = 10
            return i + 15
        return JSONResponse({'x-token': request.headers.getlist('x-token')})
    app = Starlette(routes=[Route('/', endpoint=homepage)])
    client = test_client_factory(app)
    response = client.get('/', headers=[('x-token', 'foo'), ('x-token', 'bar')])
    assert response.json() == {'x-token': ['foo', 'bar']}