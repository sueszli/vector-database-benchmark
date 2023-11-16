from __future__ import annotations
import asyncio
import json
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect
from reactpy.backend._common import ASSETS_PATH, CLIENT_BUILD_DIR, MODULES_PATH, STREAM_PATH, CommonOptions, read_client_index_html, serve_with_uvicorn
from reactpy.backend.hooks import ConnectionContext
from reactpy.backend.hooks import use_connection as _use_connection
from reactpy.backend.types import Connection, Location
from reactpy.config import REACTPY_WEB_MODULES_DIR
from reactpy.core.layout import Layout
from reactpy.core.serve import RecvCoroutine, SendCoroutine, serve_layout
from reactpy.core.types import RootComponentConstructor
logger = logging.getLogger(__name__)

@dataclass
class Options(CommonOptions):
    """Render server config for :func:`reactpy.backend.starlette.configure`"""
    cors: bool | dict[str, Any] = False
    'Enable or configure Cross Origin Resource Sharing (CORS)\n\n    For more information see docs for ``starlette.middleware.cors.CORSMiddleware``\n    '

def configure(app: Starlette, component: RootComponentConstructor, options: Options | None=None) -> None:
    if False:
        return 10
    'Configure the necessary ReactPy routes on the given app.\n\n    Parameters:\n        app: An application instance\n        component: A component constructor\n        options: Options for configuring server behavior\n    '
    options = options or Options()
    _setup_single_view_dispatcher_route(options, app, component)
    _setup_common_routes(options, app)

def create_development_app() -> Starlette:
    if False:
        i = 10
        return i + 15
    'Return a :class:`Starlette` app instance in debug mode'
    return Starlette(debug=True)

async def serve_development_app(app: Starlette, host: str, port: int, started: asyncio.Event | None=None) -> None:
    """Run a development server for starlette"""
    await serve_with_uvicorn(app, host, port, started)

def use_websocket() -> WebSocket:
    if False:
        return 10
    'Get the current WebSocket object'
    return use_connection().carrier

def use_connection() -> Connection[WebSocket]:
    if False:
        return 10
    conn = _use_connection()
    if not isinstance(conn.carrier, WebSocket):
        msg = f'Connection has unexpected carrier {conn.carrier}. Are you running with a Flask server?'
        raise TypeError(msg)
    return conn

def _setup_common_routes(options: Options, app: Starlette) -> None:
    if False:
        return 10
    cors_options = options.cors
    if cors_options:
        cors_params = cors_options if isinstance(cors_options, dict) else {'allow_origins': ['*']}
        app.add_middleware(CORSMiddleware, **cors_params)
    url_prefix = options.url_prefix
    app.mount(str(MODULES_PATH), StaticFiles(directory=REACTPY_WEB_MODULES_DIR.current, check_dir=False))
    app.mount(str(ASSETS_PATH), StaticFiles(directory=CLIENT_BUILD_DIR / 'assets', check_dir=False))
    index_route = _make_index_route(options)
    if options.serve_index_route:
        app.add_route(f'{url_prefix}/', index_route)
        app.add_route(url_prefix + '/{path:path}', index_route)

def _make_index_route(options: Options) -> Callable[[Request], Awaitable[HTMLResponse]]:
    if False:
        i = 10
        return i + 15
    index_html = read_client_index_html(options)

    async def serve_index(request: Request) -> HTMLResponse:
        return HTMLResponse(index_html)
    return serve_index

def _setup_single_view_dispatcher_route(options: Options, app: Starlette, component: RootComponentConstructor) -> None:
    if False:
        for i in range(10):
            print('nop')

    @app.websocket_route(str(STREAM_PATH))
    @app.websocket_route(f'{STREAM_PATH}/{{path:path}}')
    async def model_stream(socket: WebSocket) -> None:
        await socket.accept()
        (send, recv) = _make_send_recv_callbacks(socket)
        pathname = '/' + socket.scope['path_params'].get('path', '')
        pathname = pathname[len(options.url_prefix):] or '/'
        search = socket.scope['query_string'].decode()
        try:
            await serve_layout(Layout(ConnectionContext(component(), value=Connection(scope=socket.scope, location=Location(pathname, f'?{search}' if search else ''), carrier=socket))), send, recv)
        except WebSocketDisconnect as error:
            logger.info(f'WebSocket disconnect: {error.code}')

def _make_send_recv_callbacks(socket: WebSocket) -> tuple[SendCoroutine, RecvCoroutine]:
    if False:
        return 10

    async def sock_send(value: Any) -> None:
        await socket.send_text(json.dumps(value))

    async def sock_recv() -> Any:
        return json.loads(await socket.receive_text())
    return (sock_send, sock_recv)