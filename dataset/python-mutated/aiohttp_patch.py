from asyncio import CancelledError, Task
from threading import Lock
from typing import Type
from aiohttp import StreamReader
from aiohttp.abc import AbstractStreamWriter
from aiohttp.http_parser import RawRequestMessage
from aiohttp.web_app import Application
from aiohttp.web_protocol import RequestHandler
from aiohttp.web_request import Request
transport_is_none_counter = 0
counter_lock = Lock()

def increment_transport_is_none_counter():
    if False:
        return 10
    global transport_is_none_counter
    with counter_lock:
        transport_is_none_counter += 1

def get_transport_is_none_counter() -> int:
    if False:
        while True:
            i = 10
    with counter_lock:
        return transport_is_none_counter

def patch_make_request(cls: Type[Application]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    original_make_request = cls._make_request
    if getattr(original_make_request, 'patched', False):
        return False

    def new_make_request(self, message: RawRequestMessage, payload: StreamReader, protocol: RequestHandler, writer: AbstractStreamWriter, task: Task, _cls: Type[Request]=Request) -> Request:
        if False:
            print('Hello World!')
        if protocol.transport is None:
            increment_transport_is_none_counter()
            raise CancelledError
        return original_make_request(self, message=message, payload=payload, protocol=protocol, writer=writer, task=task, _cls=_cls)
    new_make_request.patched = True
    cls._make_request = new_make_request
    return True