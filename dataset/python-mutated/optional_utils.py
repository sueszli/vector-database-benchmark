"""
Optional utils module contains utility methods
that require optional dependencies.
"""
import asyncio
import collections
import functools
import inspect
import json
import logging
import os
import time
import traceback
from collections import namedtuple
from typing import Any, Callable
from aiohttp.web import Response
import ray
import ray.dashboard.consts as dashboard_consts
from ray._private.ray_constants import RAY_INTERNAL_DASHBOARD_NAMESPACE, env_bool
from ray.dashboard.optional_deps import PathLike, RouteDef, aiohttp, hdrs
from ray.dashboard.utils import CustomEncoder, to_google_style
try:
    create_task = asyncio.create_task
except AttributeError:
    create_task = asyncio.ensure_future
logger = logging.getLogger(__name__)

def method_route_table_factory():
    if False:
        return 10

    class MethodRouteTable:
        """A helper class to bind http route to class method."""
        _bind_map = collections.defaultdict(dict)
        _routes = aiohttp.web.RouteTableDef()

        class _BindInfo:

            def __init__(self, filename, lineno, instance):
                if False:
                    return 10
                self.filename = filename
                self.lineno = lineno
                self.instance = instance

        @classmethod
        def routes(cls):
            if False:
                return 10
            return cls._routes

        @classmethod
        def bound_routes(cls):
            if False:
                while True:
                    i = 10
            bound_items = []
            for r in cls._routes._items:
                if isinstance(r, RouteDef):
                    route_method = getattr(r.handler, '__route_method__')
                    route_path = getattr(r.handler, '__route_path__')
                    instance = cls._bind_map[route_method][route_path].instance
                    if instance is not None:
                        bound_items.append(r)
                else:
                    bound_items.append(r)
            routes = aiohttp.web.RouteTableDef()
            routes._items = bound_items
            return routes

        @classmethod
        def _register_route(cls, method, path, **kwargs):
            if False:
                i = 10
                return i + 15

            def _wrapper(handler):
                if False:
                    for i in range(10):
                        print('nop')
                if path in cls._bind_map[method]:
                    bind_info = cls._bind_map[method][path]
                    raise Exception(f'Duplicated route path: {path}, previous one registered at {bind_info.filename}:{bind_info.lineno}')
                bind_info = cls._BindInfo(handler.__code__.co_filename, handler.__code__.co_firstlineno, None)

                @functools.wraps(handler)
                async def _handler_route(*args) -> aiohttp.web.Response:
                    try:
                        req = args[-1]
                        return await handler(bind_info.instance, req)
                    except Exception:
                        logger.exception('Handle %s %s failed.', method, path)
                        return rest_response(success=False, message=traceback.format_exc())
                cls._bind_map[method][path] = bind_info
                _handler_route.__route_method__ = method
                _handler_route.__route_path__ = path
                return cls._routes.route(method, path, **kwargs)(_handler_route)
            return _wrapper

        @classmethod
        def head(cls, path, **kwargs):
            if False:
                return 10
            return cls._register_route(hdrs.METH_HEAD, path, **kwargs)

        @classmethod
        def get(cls, path, **kwargs):
            if False:
                return 10
            return cls._register_route(hdrs.METH_GET, path, **kwargs)

        @classmethod
        def post(cls, path, **kwargs):
            if False:
                i = 10
                return i + 15
            return cls._register_route(hdrs.METH_POST, path, **kwargs)

        @classmethod
        def put(cls, path, **kwargs):
            if False:
                return 10
            return cls._register_route(hdrs.METH_PUT, path, **kwargs)

        @classmethod
        def patch(cls, path, **kwargs):
            if False:
                return 10
            return cls._register_route(hdrs.METH_PATCH, path, **kwargs)

        @classmethod
        def delete(cls, path, **kwargs):
            if False:
                i = 10
                return i + 15
            return cls._register_route(hdrs.METH_DELETE, path, **kwargs)

        @classmethod
        def view(cls, path, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return cls._register_route(hdrs.METH_ANY, path, **kwargs)

        @classmethod
        def static(cls, prefix: str, path: PathLike, **kwargs: Any) -> None:
            if False:
                while True:
                    i = 10
            cls._routes.static(prefix, path, **kwargs)

        @classmethod
        def bind(cls, instance):
            if False:
                i = 10
                return i + 15

            def predicate(o):
                if False:
                    for i in range(10):
                        print('nop')
                if inspect.ismethod(o):
                    return hasattr(o, '__route_method__') and hasattr(o, '__route_path__')
                return False
            handler_routes = inspect.getmembers(instance, predicate)
            for (_, h) in handler_routes:
                cls._bind_map[h.__func__.__route_method__][h.__func__.__route_path__].instance = instance
    return MethodRouteTable
DashboardHeadRouteTable = method_route_table_factory()
DashboardAgentRouteTable = method_route_table_factory()

def rest_response(success, message, convert_google_style=True, reason=None, **kwargs) -> aiohttp.web.Response:
    if False:
        for i in range(10):
            print('nop')
    if os.environ.get('RAY_DASHBOARD_DEV') == '1':
        headers = {'Access-Control-Allow-Origin': '*'}
    else:
        headers = {}
    return aiohttp.web.json_response({'result': success, 'msg': message, 'data': to_google_style(kwargs) if convert_google_style else kwargs}, dumps=functools.partial(json.dumps, cls=CustomEncoder), headers=headers, status=200 if success else 500, reason=reason)
_AiohttpCacheValue = namedtuple('AiohttpCacheValue', ['data', 'expiration', 'task'])
_AIOHTTP_CACHE_NOBODY_METHODS = {hdrs.METH_GET, hdrs.METH_DELETE}

def aiohttp_cache(ttl_seconds=dashboard_consts.AIOHTTP_CACHE_TTL_SECONDS, maxsize=dashboard_consts.AIOHTTP_CACHE_MAX_SIZE, enable=not env_bool(dashboard_consts.AIOHTTP_CACHE_DISABLE_ENVIRONMENT_KEY, False)):
    if False:
        for i in range(10):
            print('nop')
    assert maxsize > 0
    cache = collections.OrderedDict()

    def _wrapper(handler):
        if False:
            print('Hello World!')
        if enable:

            @functools.wraps(handler)
            async def _cache_handler(*args) -> aiohttp.web.Response:
                req = args[-1]
                if req.method in _AIOHTTP_CACHE_NOBODY_METHODS:
                    key = req.path_qs
                else:
                    key = (req.path_qs, await req.read())
                value = cache.get(key)
                if value is not None:
                    cache.move_to_end(key)
                    if not value.task.done() or value.expiration >= time.time():
                        return aiohttp.web.Response(**value.data)

                def _update_cache(task):
                    if False:
                        for i in range(10):
                            print('nop')
                    try:
                        response = task.result()
                    except Exception:
                        response = rest_response(success=False, message=traceback.format_exc())
                    data = {'status': response.status, 'headers': dict(response.headers), 'body': response.body}
                    cache[key] = _AiohttpCacheValue(data, time.time() + ttl_seconds, task)
                    cache.move_to_end(key)
                    if len(cache) > maxsize:
                        cache.popitem(last=False)
                    return response
                task = create_task(handler(*args))
                task.add_done_callback(_update_cache)
                if value is None:
                    return await task
                else:
                    return aiohttp.web.Response(**value.data)
            suffix = f'[cache ttl={ttl_seconds}, max_size={maxsize}]'
            _cache_handler.__name__ += suffix
            _cache_handler.__qualname__ += suffix
            return _cache_handler
        else:
            return handler
    if inspect.iscoroutinefunction(ttl_seconds):
        target_func = ttl_seconds
        ttl_seconds = dashboard_consts.AIOHTTP_CACHE_TTL_SECONDS
        return _wrapper(target_func)
    else:
        return _wrapper

def init_ray_and_catch_exceptions() -> Callable:
    if False:
        print('Hello World!')
    'Decorator to be used on methods that require being connected to Ray.'

    def decorator_factory(f: Callable) -> Callable:
        if False:
            i = 10
            return i + 15

        @functools.wraps(f)
        async def decorator(self, *args, **kwargs):
            try:
                if not ray.is_initialized():
                    try:
                        address = self.get_gcs_address()
                        logger.info(f'Connecting to ray with address={address}')
                        os.environ['RAY_gcs_server_request_timeout_seconds'] = str(dashboard_consts.GCS_RPC_TIMEOUT_SECONDS)
                        ray.init(address=address, log_to_driver=False, configure_logging=False, namespace=RAY_INTERNAL_DASHBOARD_NAMESPACE, _skip_env_hook=True)
                    except Exception as e:
                        ray.shutdown()
                        raise e from None
                return await f(self, *args, **kwargs)
            except Exception as e:
                logger.exception(f'Unexpected error in handler: {e}')
                return Response(text=traceback.format_exc(), status=aiohttp.web.HTTPInternalServerError.status_code)
        return decorator
    return decorator_factory