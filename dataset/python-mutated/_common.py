from __future__ import annotations
import asyncio
import os
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast
from reactpy import __file__ as _reactpy_file_path
from reactpy import html
from reactpy.config import REACTPY_WEB_MODULES_DIR
from reactpy.core.types import VdomDict
from reactpy.utils import vdom_to_html
if TYPE_CHECKING:
    import uvicorn
    from asgiref.typing import ASGIApplication
PATH_PREFIX = PurePosixPath('/_reactpy')
MODULES_PATH = PATH_PREFIX / 'modules'
ASSETS_PATH = PATH_PREFIX / 'assets'
STREAM_PATH = PATH_PREFIX / 'stream'
CLIENT_BUILD_DIR = Path(_reactpy_file_path).parent / '_static' / 'app' / 'dist'

async def serve_with_uvicorn(app: ASGIApplication | Any, host: str, port: int, started: asyncio.Event | None) -> None:
    """Run a development server for an ASGI application"""
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, loop='asyncio'))
    server.config.setup_event_loop()
    coros: list[Awaitable[Any]] = [server.serve()]
    if started:
        coros.append(_check_if_started(server, started))
    try:
        await asyncio.gather(*coros)
    finally:
        if not hasattr(server, 'servers'):
            server.servers = []
        await asyncio.wait_for(server.shutdown(), timeout=3)

async def _check_if_started(server: uvicorn.Server, started: asyncio.Event) -> None:
    while not server.started:
        await asyncio.sleep(0.2)
    started.set()

def safe_client_build_dir_path(path: str) -> Path:
    if False:
        while True:
            i = 10
    'Prevent path traversal out of :data:`CLIENT_BUILD_DIR`'
    return traversal_safe_path(CLIENT_BUILD_DIR, *('index.html' if path in {'', '/'} else path).split('/'))

def safe_web_modules_dir_path(path: str) -> Path:
    if False:
        while True:
            i = 10
    'Prevent path traversal out of :data:`reactpy.config.REACTPY_WEB_MODULES_DIR`'
    return traversal_safe_path(REACTPY_WEB_MODULES_DIR.current, *path.split('/'))

def traversal_safe_path(root: str | Path, *unsafe: str | Path) -> Path:
    if False:
        for i in range(10):
            print('nop')
    'Raise a ``ValueError`` if the ``unsafe`` path resolves outside the root dir.'
    root = os.path.abspath(root)
    path = os.path.abspath(os.path.join(root, *unsafe))
    if os.path.commonprefix([root, path]) != root:
        msg = 'Unsafe path'
        raise ValueError(msg)
    return Path(path)

def read_client_index_html(options: CommonOptions) -> str:
    if False:
        while True:
            i = 10
    return (CLIENT_BUILD_DIR / 'index.html').read_text().format(__head__=vdom_head_elements_to_html(options.head))

def vdom_head_elements_to_html(head: Sequence[VdomDict] | VdomDict | str) -> str:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(head, str):
        return head
    elif isinstance(head, dict):
        if head.get('tagName') == 'head':
            head = cast(VdomDict, {**head, 'tagName': ''})
        return vdom_to_html(head)
    else:
        return vdom_to_html(html._(*head))

@dataclass
class CommonOptions:
    """Options for ReactPy's built-in backed server implementations"""
    head: Sequence[VdomDict] | VdomDict | str = (html.title('ReactPy'), html.link({'rel': 'icon', 'href': '/_reactpy/assets/reactpy-logo.ico', 'type': 'image/x-icon'}))
    'Add elements to the ``<head>`` of the application.\n\n    For example, this can be used to customize the title of the page, link extra\n    scripts, or load stylesheets.\n    '
    url_prefix: str = ''
    'The URL prefix where ReactPy resources will be served from'
    serve_index_route: bool = True
    'Automatically generate and serve the index route (``/``)'

    def __post_init__(self) -> None:
        if False:
            while True:
                i = 10
        if self.url_prefix and (not self.url_prefix.startswith('/')):
            msg = "Expected 'url_prefix' to start with '/'"
            raise ValueError(msg)