"""Utilities for the App class."""
from inspect import iscoroutinefunction
from typing import IO, Iterable, List, Tuple
from falcon import util
from falcon.constants import MEDIA_JSON
from falcon.constants import MEDIA_XML
from falcon.errors import CompatibilityError, HTTPError
from falcon.request import Request
from falcon.response import Response
from falcon.util.sync import _wrap_non_coroutine_unsafe
__all__ = ('prepare_middleware', 'prepare_middleware_ws', 'default_serialize_error', 'CloseableStreamIterator')

def prepare_middleware(middleware: Iterable, independent_middleware: bool=False, asgi: bool=False) -> Tuple[tuple, tuple, tuple]:
    if False:
        while True:
            i = 10
    'Check middleware interfaces and prepare the methods for request handling.\n\n    Note:\n        This method is only applicable to WSGI apps.\n\n    Arguments:\n        middleware (iterable): An iterable of middleware objects.\n\n    Keyword Args:\n        independent_middleware (bool): ``True`` if the request and\n            response middleware methods should be treated independently\n            (default ``False``)\n        asgi (bool): ``True`` if an ASGI app, ``False`` otherwise\n            (default ``False``)\n\n    Returns:\n        tuple: A tuple of prepared middleware method tuples\n    '
    request_mw: List = []
    resource_mw: List = []
    response_mw: List = []
    for component in middleware:
        if asgi:
            process_request = util.get_bound_method(component, 'process_request_async') or _wrap_non_coroutine_unsafe(util.get_bound_method(component, 'process_request'))
            process_resource = util.get_bound_method(component, 'process_resource_async') or _wrap_non_coroutine_unsafe(util.get_bound_method(component, 'process_resource'))
            process_response = util.get_bound_method(component, 'process_response_async') or _wrap_non_coroutine_unsafe(util.get_bound_method(component, 'process_response'))
            for m in (process_request, process_resource, process_response):
                if m and (not iscoroutinefunction(m)) and util.is_python_func(m):
                    msg = '{} must be implemented as an awaitable coroutine. If you would like to retain compatibility with WSGI apps, the coroutine versions of the middleware methods may be implemented side-by-side by applying an *_async postfix to the method names. '
                    raise CompatibilityError(msg.format(m))
        else:
            process_request = util.get_bound_method(component, 'process_request')
            process_resource = util.get_bound_method(component, 'process_resource')
            process_response = util.get_bound_method(component, 'process_response')
            for m in (process_request, process_resource, process_response):
                if m and iscoroutinefunction(m):
                    msg = '{} may not implement coroutine methods and remain compatible with WSGI apps without using the *_async postfix to explicitly identify the coroutine version of a given middleware method.'
                    raise CompatibilityError(msg.format(component))
        if not (process_request or process_resource or process_response):
            if asgi and any((hasattr(component, m) for m in ['process_startup', 'process_shutdown', 'process_request_ws', 'process_resource_ws'])):
                continue
            msg = '{0} must implement at least one middleware method'
            raise TypeError(msg.format(component))
        if independent_middleware:
            if process_request:
                request_mw.append(process_request)
            if process_response:
                response_mw.insert(0, process_response)
        elif process_request or process_response:
            request_mw.append((process_request, process_response))
        if process_resource:
            resource_mw.append(process_resource)
    return (tuple(request_mw), tuple(resource_mw), tuple(response_mw))

def prepare_middleware_ws(middleware: Iterable) -> Tuple[list, list]:
    if False:
        i = 10
        return i + 15
    'Check middleware interfaces and prepare WebSocket methods for request handling.\n\n    Note:\n        This method is only applicable to ASGI apps.\n\n    Arguments:\n        middleware (iterable): An iterable of middleware objects.\n\n    Returns:\n        tuple: A two-item ``(request_mw, resource_mw)`` tuple, where\n        *request_mw* is an ordered list of ``process_request_ws()`` methods,\n        and *resource_mw* is an ordered list of ``process_resource_ws()``\n        methods.\n    '
    request_mw = []
    resource_mw = []
    for component in middleware:
        process_request_ws = util.get_bound_method(component, 'process_request_ws')
        process_resource_ws = util.get_bound_method(component, 'process_resource_ws')
        for m in (process_request_ws, process_resource_ws):
            if not m:
                continue
            if not iscoroutinefunction(m) and util.is_python_func(m):
                msg = '{} must be implemented as an awaitable coroutine.'
                raise CompatibilityError(msg.format(m))
        if process_request_ws:
            request_mw.append(process_request_ws)
        if process_resource_ws:
            resource_mw.append(process_resource_ws)
    return (request_mw, resource_mw)

def default_serialize_error(req: Request, resp: Response, exception: HTTPError):
    if False:
        while True:
            i = 10
    'Serialize the given instance of HTTPError.\n\n    This function determines which of the supported media types, if\n    any, are acceptable by the client, and serializes the error\n    to the preferred type.\n\n    Currently, JSON and XML are the only supported media types. If the\n    client accepts both JSON and XML with equal weight, JSON will be\n    chosen.\n\n    Other media types can be supported by using a custom error serializer.\n\n    Note:\n        If a custom media type is used and the type includes a\n        "+json" or "+xml" suffix, the error will be serialized\n        to JSON or XML, respectively. If this behavior is not\n        desirable, a custom error serializer may be used to\n        override this one.\n\n    Args:\n        req: Instance of ``falcon.Request``\n        resp: Instance of ``falcon.Response``\n        exception: Instance of ``falcon.HTTPError``\n    '
    preferred = req.client_prefers((MEDIA_XML, 'text/xml', MEDIA_JSON))
    if preferred is None:
        accept = req.accept.lower()
        if '+json' in accept:
            preferred = MEDIA_JSON
        elif '+xml' in accept:
            preferred = MEDIA_XML
    if preferred is not None:
        if preferred == MEDIA_JSON:
            (handler, _, _) = resp.options.media_handlers._resolve(MEDIA_JSON, MEDIA_JSON, raise_not_found=False)
            resp.data = exception.to_json(handler)
        else:
            resp.data = exception.to_xml()
        resp.content_type = preferred
    resp.append_header('Vary', 'Accept')

class CloseableStreamIterator:
    """Iterator that wraps a file-like stream with support for close().

    This iterator can be used to read from an underlying file-like stream
    in block_size-chunks until the response from the stream is an empty
    byte string.

    This class is used to wrap WSGI response streams when a
    wsgi_file_wrapper is not provided by the server.  The fact that it
    also supports closing the underlying stream allows use of (e.g.)
    Python tempfile resources that would be deleted upon close.

    Args:
        stream (object): Readable file-like stream object.
        block_size (int): Number of bytes to read per iteration.
    """

    def __init__(self, stream: IO, block_size: int):
        if False:
            return 10
        self._stream = stream
        self._block_size = block_size

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        data = self._stream.read(self._block_size)
        if data == b'':
            raise StopIteration
        else:
            return data

    def close(self):
        if False:
            while True:
                i = 10
        try:
            self._stream.close()
        except (AttributeError, TypeError):
            pass