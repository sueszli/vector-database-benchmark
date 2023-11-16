from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING
from sanic.exceptions import SanicException
if TYPE_CHECKING:
    from sanic import Sanic

class AsyncioServer:
    """Wraps an asyncio server with functionality that might be useful to a user who needs to manage the server lifecycle manually."""
    __slots__ = ('app', 'connections', 'loop', 'serve_coro', 'server')

    def __init__(self, app: Sanic, loop, serve_coro, connections):
        if False:
            while True:
                i = 10
        self.app = app
        self.connections = connections
        self.loop = loop
        self.serve_coro = serve_coro
        self.server = None

    def startup(self):
        if False:
            for i in range(10):
                print('nop')
        'Trigger "startup" operations on the app'
        return self.app._startup()

    def before_start(self):
        if False:
            for i in range(10):
                print('nop')
        'Trigger "before_server_start" events'
        return self._server_event('init', 'before')

    def after_start(self):
        if False:
            return 10
        'Trigger "after_server_start" events'
        return self._server_event('init', 'after')

    def before_stop(self):
        if False:
            while True:
                i = 10
        'Trigger "before_server_stop" events'
        return self._server_event('shutdown', 'before')

    def after_stop(self):
        if False:
            print('Hello World!')
        'Trigger "after_server_stop" events'
        return self._server_event('shutdown', 'after')

    def is_serving(self) -> bool:
        if False:
            print('Hello World!')
        'Returns True if the server is running, False otherwise'
        if self.server:
            return self.server.is_serving()
        return False

    def wait_closed(self):
        if False:
            return 10
        'Wait until the server is closed'
        if self.server:
            return self.server.wait_closed()

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        'Close the server'
        if self.server:
            self.server.close()
            coro = self.wait_closed()
            task = asyncio.ensure_future(coro, loop=self.loop)
            return task

    def start_serving(self):
        if False:
            print('Hello World!')
        'Start serving requests'
        return self._serve(self.server.start_serving)

    def serve_forever(self):
        if False:
            for i in range(10):
                print('nop')
        'Serve requests until the server is stopped'
        return self._serve(self.server.serve_forever)

    def _serve(self, serve_func):
        if False:
            print('Hello World!')
        if self.server:
            if not self.app.state.is_started:
                raise SanicException('Cannot run Sanic server without first running await server.startup()')
            try:
                return serve_func()
            except AttributeError:
                name = serve_func.__name__
                raise NotImplementedError(f'server.{name} not available in this version of asyncio or uvloop.')

    def _server_event(self, concern: str, action: str):
        if False:
            while True:
                i = 10
        if not self.app.state.is_started:
            raise SanicException('Cannot dispatch server event without first running await server.startup()')
        return self.app._server_event(concern, action, loop=self.loop)

    def __await__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Starts the asyncio server, returns AsyncServerCoro\n        '
        task = asyncio.ensure_future(self.serve_coro)
        while not task.done():
            yield
        self.server = task.result()
        return self