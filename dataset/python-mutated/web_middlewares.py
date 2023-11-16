import re
import warnings
from typing import TYPE_CHECKING, Tuple, Type, TypeVar
from .typedefs import Handler, Middleware
from .web_exceptions import HTTPMove, HTTPPermanentRedirect
from .web_request import Request
from .web_response import StreamResponse
from .web_urldispatcher import SystemRoute
__all__ = ('middleware', 'normalize_path_middleware')
if TYPE_CHECKING:
    from .web_app import Application
_Func = TypeVar('_Func')

async def _check_request_resolves(request: Request, path: str) -> Tuple[bool, Request]:
    alt_request = request.clone(rel_url=path)
    match_info = await request.app.router.resolve(alt_request)
    alt_request._match_info = match_info
    if match_info.http_exception is None:
        return (True, alt_request)
    return (False, request)

def middleware(f: _Func) -> _Func:
    if False:
        i = 10
        return i + 15
    warnings.warn('Middleware decorator is deprecated since 4.0 and its behaviour is default, you can simply remove this decorator.', DeprecationWarning, stacklevel=2)
    return f

def normalize_path_middleware(*, append_slash: bool=True, remove_slash: bool=False, merge_slashes: bool=True, redirect_class: Type[HTTPMove]=HTTPPermanentRedirect) -> Middleware:
    if False:
        print('Hello World!')
    'Factory for producing a middleware that normalizes the path of a request.\n\n    Normalizing means:\n        - Add or remove a trailing slash to the path.\n        - Double slashes are replaced by one.\n\n    The middleware returns as soon as it finds a path that resolves\n    correctly. The order if both merge and append/remove are enabled is\n        1) merge slashes\n        2) append/remove slash\n        3) both merge slashes and append/remove slash.\n    If the path resolves with at least one of those conditions, it will\n    redirect to the new path.\n\n    Only one of `append_slash` and `remove_slash` can be enabled. If both\n    are `True` the factory will raise an assertion error\n\n    If `append_slash` is `True` the middleware will append a slash when\n    needed. If a resource is defined with trailing slash and the request\n    comes without it, it will append it automatically.\n\n    If `remove_slash` is `True`, `append_slash` must be `False`. When enabled\n    the middleware will remove trailing slashes and redirect if the resource\n    is defined\n\n    If merge_slashes is True, merge multiple consecutive slashes in the\n    path into one.\n    '
    correct_configuration = not (append_slash and remove_slash)
    assert correct_configuration, 'Cannot both remove and append slash'

    async def impl(request: Request, handler: Handler) -> StreamResponse:
        if isinstance(request.match_info.route, SystemRoute):
            paths_to_check = []
            if '?' in request.raw_path:
                (path, query) = request.raw_path.split('?', 1)
                query = '?' + query
            else:
                query = ''
                path = request.raw_path
            if merge_slashes:
                paths_to_check.append(re.sub('//+', '/', path))
            if append_slash and (not request.path.endswith('/')):
                paths_to_check.append(path + '/')
            if remove_slash and request.path.endswith('/'):
                paths_to_check.append(path[:-1])
            if merge_slashes and append_slash:
                paths_to_check.append(re.sub('//+', '/', path + '/'))
            if merge_slashes and remove_slash and path.endswith('/'):
                merged_slashes = re.sub('//+', '/', path)
                paths_to_check.append(merged_slashes[:-1])
            for path in paths_to_check:
                path = re.sub('^//+', '/', path)
                (resolves, request) = await _check_request_resolves(request, path)
                if resolves:
                    raise redirect_class(request.raw_path + query)
        return await handler(request)
    return impl

def _fix_request_current_app(app: 'Application') -> Middleware:
    if False:
        return 10

    async def impl(request: Request, handler: Handler) -> StreamResponse:
        with request.match_info.set_current_app(app):
            return await handler(request)
    return impl