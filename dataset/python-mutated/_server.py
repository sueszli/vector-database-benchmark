"""
High level code related to the server that provides a mainloop and
serves the pages and websocket.
"""
import sys
import asyncio
from ..event import _loop
from .. import config
from . import logger
_current_server = None

def create_server(host=None, port=None, loop=None, backend='tornado', **server_kwargs):
    if False:
        print('Hello World!')
    "\n    Create a new server object. This is automatically called; users generally\n    don't need this, unless they want to explicitly specify host/port,\n    create a fresh server in testing scenarios, or run Flexx in a thread.\n\n    Flexx uses the notion of a single current server object. This function\n    (re)creates that object. If there already was a server object, it is\n    replaced. It is an error to call this function if the current server\n    is still running.\n\n    Arguments:\n        host (str): The hostname to serve on. By default\n            ``flexx.config.hostname`` is used. If ``False``, do not listen\n            (e.g. when integrating with an existing Tornado application).\n        port (int, str): The port number. If a string is given, it is\n            hashed to an ephemeral port number. By default\n            ``flexx.config.port`` is used.\n        loop: A fresh (asyncio) event loop, default None (use current).\n        backend (str): Stub argument; only Tornado is currently supported.\n        **server_kwargs: keyword arguments passed to the server constructor.\n\n    Returns:\n        AbstractServer: The server object, see ``current_server()``.\n    "
    global _current_server
    if host is None:
        host = config.hostname
    if port is None:
        port = config.port
    if _current_server:
        _current_server.close()
    backend = backend.lower()
    if backend == 'tornado':
        from ._tornadoserver import TornadoServer
        _current_server = TornadoServer(host, port, loop, **server_kwargs)
    elif backend == 'flask':
        from ._flaskserver import FlaskServer
        _current_server = FlaskServer(host, port, loop, **server_kwargs)
    else:
        raise RuntimeError('Flexx server can only run on Tornado and Flask (for now).')
    assert isinstance(_current_server, AbstractServer)
    return _current_server

def current_server(create=True, **server_kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Get the current server object. Creates a server if there is none\n    and the ``create`` arg is True. Currently, this is always a\n    TornadoServer object, which has properties:\n\n    * serving: a tuple ``(hostname, port)`` specifying the location\n      being served (or ``None`` if the server is closed).\n    * protocol: the protocol (e.g. "http") being used.\n    * app: the ``tornado.web.Application`` instance\n    * server: the ``tornado.httpserver.HttpServer`` instance\n\n    '
    if create and (not _current_server):
        create_server(**server_kwargs)
    return _current_server

async def keep_awake():
    while True:
        await asyncio.sleep(0.2)

class AbstractServer:
    """ This is an attempt to generalize the server, so that in the
    future we may have e.g. a Flask or Pyramid server.

    A server must implement this, and use the manager to instantiate,
    connect and disconnect sessions. The assets object must be used to
    server assets to the client.

    Arguments:
        host (str): the hostname to serve at
        port (int): the port to serve at. None or 0 mean to autoselect a port.
    """

    def __init__(self, host, port, loop=None, **kwargs):
        if False:
            while True:
                i = 10
        if sys.version_info > (3, 8) and sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        if loop is None:
            self._loop = asyncio.get_event_loop()
        else:
            assert isinstance(loop, asyncio.AbstractEventLoop)
            self._loop = loop
        asyncio.set_event_loop(self._loop)
        _loop.loop.integrate(self._loop, reset=False)
        self._serving = None
        if host is not False:
            self._open(host, port, **kwargs)
            assert self._serving

    @property
    def _running(self):
        if False:
            i = 10
            return i + 15
        return self._loop.is_running()

    def start(self):
        if False:
            return 10
        ' Start the event loop. '
        if not self._serving:
            raise RuntimeError('Cannot start a closed or non-serving server!')
        if self._running:
            raise RuntimeError('Cannot start a running server.')
        if asyncio.get_event_loop() is not self._loop:
            raise RuntimeError('Can only start server in same thread that created it.')
        logger.info('Starting Flexx event loop.')
        if not getattr(self._loop, '_in_event_loop', False):
            poller = self._loop.create_task(keep_awake())
            try:
                self._loop.run_forever()
            except KeyboardInterrupt:
                logger.info('Flexx event loop interrupted.')
            except TypeError as err:
                if 'close() takes 1 positional argument but 3 were given' in str(err):
                    logger.info('Interrupted Flexx event loop.')
                else:
                    raise
            poller.cancel()

    def stop(self):
        if False:
            return 10
        ' Stop the event loop. This does not close the connection; the server\n        can be restarted. Thread safe. '
        logger.info('Stopping Flexx event loop.')
        self._loop.call_soon_threadsafe(self._loop.stop)

    def close(self):
        if False:
            i = 10
            return i + 15
        ' Close the connection. A closed server cannot be used again. '
        if self._running:
            raise RuntimeError('Cannot close a running server; need to stop first.')
        self._serving = None
        self._close()

    def _open(self, host, port, **kwargs):
        if False:
            return 10
        raise NotImplementedError()

    def _close(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @property
    def serving(self):
        if False:
            for i in range(10):
                print('nop')
        ' Get a tuple (hostname, port) that is being served.\n        Or None if the server is not serving (anymore).\n        '
        return self._serving

    @property
    def protocol(self):
        if False:
            while True:
                i = 10
        ' Get a string representing served protocol\n        '
        raise NotImplementedError