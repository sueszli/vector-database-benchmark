"""Helper to restore old aiohttp behavior."""
from __future__ import annotations
from aiohttp import web, web_protocol, web_server

class CancelOnDisconnectRequestHandler(web_protocol.RequestHandler):
    """Request handler that cancels tasks on disconnect."""

    def connection_lost(self, exc: BaseException | None) -> None:
        if False:
            return 10
        'Handle connection lost.'
        task_handler = self._task_handler
        super().connection_lost(exc)
        if task_handler is not None:
            task_handler.cancel('aiohttp connection lost')

def restore_original_aiohttp_cancel_behavior() -> None:
    if False:
        i = 10
        return i + 15
    'Patch aiohttp to restore cancel behavior.\n\n    Remove this once aiohttp 3.9 is released as we can use\n    https://github.com/aio-libs/aiohttp/pull/7128\n    '
    web_protocol.RequestHandler = CancelOnDisconnectRequestHandler
    web_server.RequestHandler = CancelOnDisconnectRequestHandler

def enable_compression(response: web.Response) -> None:
    if False:
        return 10
    'Enable compression on the response.'
    response._zlib_executor_size = 32768
    response.enable_compression()