"""The watch_app_state function enables us to trigger a callback function when ever the app state changes."""
from __future__ import annotations
import asyncio
import os
from threading import Thread
from typing import Callable
import websockets
from lightning.app.core.constants import APP_SERVER_PORT
from lightning.app.utilities.app_helpers import Logger
_logger = Logger(__name__)
_CALLBACKS = []
_THREAD: Thread = None

def _get_ws_port():
    if False:
        while True:
            i = 10
    if 'LIGHTNING_APP_STATE_URL' in os.environ:
        return 8080
    return APP_SERVER_PORT

def _get_ws_url():
    if False:
        for i in range(10):
            print('nop')
    port = _get_ws_port()
    return f'ws://localhost:{port}/api/v1/ws'

def _run_callbacks():
    if False:
        while True:
            i = 10
    for callback in _CALLBACKS:
        callback()

def _target_fn():
    if False:
        return 10

    async def update_fn():
        ws_url = _get_ws_url()
        _logger.debug('connecting to web socket %s', ws_url)
        async with websockets.connect(ws_url) as websocket:
            while True:
                await websocket.recv()
                _logger.debug('App State Changed. Running callbacks')
                _run_callbacks()
    asyncio.run(update_fn())

def _start_websocket():
    if False:
        print('Hello World!')
    global _THREAD
    if not _THREAD:
        _logger.debug('Starting the watch_app_state thread.')
        _THREAD = Thread(target=_target_fn)
        _THREAD.setDaemon(True)
        _THREAD.start()
        _logger.debug('thread started')

def _watch_app_state(callback: Callable):
    if False:
        for i in range(10):
            print('nop')
    'Start the process that serves the UI at the given hostname and port number.\n\n    Arguments:\n        callback: A function to run when the App state changes. Must be thread safe.\n\n    Example:\n\n        .. code-block:: python\n\n            def handle_state_change():\n                print("The App State changed.")\n                watch_app_state(handle_state_change)\n\n    '
    _CALLBACKS.append(callback)
    _start_websocket()