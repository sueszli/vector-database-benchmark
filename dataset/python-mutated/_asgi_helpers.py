import functools
import inspect
from falcon.errors import UnsupportedError, UnsupportedScopeError

@functools.lru_cache(maxsize=16)
def _validate_asgi_scope(scope_type, spec_version, http_version):
    if False:
        while True:
            i = 10
    if scope_type == 'http':
        spec_version = spec_version or '2.0'
        if not spec_version.startswith('2.'):
            raise UnsupportedScopeError(f'The ASGI "http" scope version {spec_version} is not supported.')
        if http_version not in {'1.0', '1.1', '2', '3'}:
            raise UnsupportedError(f'The ASGI "http" scope does not support HTTP version {http_version}.')
        return spec_version
    if scope_type == 'websocket':
        spec_version = spec_version or '2.0'
        if not spec_version.startswith('2.'):
            raise UnsupportedScopeError('Only versions 2.x of the ASGI "websocket" scope are supported.')
        if http_version not in {'1.1', '2', '3'}:
            raise UnsupportedError(f'The ASGI "websocket" scope does not support HTTP version {http_version}.')
        return spec_version
    if scope_type == 'lifespan':
        spec_version = spec_version or '1.0'
        if not spec_version.startswith('1.') and (not spec_version.startswith('2.')):
            raise UnsupportedScopeError('Only versions 1.x and 2.x of the ASGI "lifespan" scope are supported.')
        return spec_version
    raise UnsupportedScopeError(f'The ASGI "{scope_type}" scope type is not supported.')

def _wrap_asgi_coroutine_func(asgi_impl):
    if False:
        i = 10
        return i + 15
    'Wrap an ASGI application in another coroutine.\n\n    This utility is used to wrap the cythonized ``App.__call__`` in order to\n    masquerade it as a pure-Python coroutine function.\n\n    Conversely, if the ASGI callable is not detected as a coroutine function,\n    the application server might incorrectly assume an ASGI 2.0 application\n    (i.e., the double-callable style).\n\n    In case the app class is not cythonized, this function is a simple\n    passthrough of the original implementation.\n\n    Args:\n        asgi_impl(callable): An ASGI application class method.\n\n    Returns:\n        A pure-Python ``__call__`` implementation.\n    '

    async def __call__(self, scope, receive, send):
        await asgi_impl(self, scope, receive, send)
    if inspect.iscoroutinefunction(asgi_impl):
        return asgi_impl
    return __call__