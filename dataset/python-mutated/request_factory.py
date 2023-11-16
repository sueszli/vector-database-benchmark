from __future__ import annotations
import json
from functools import partial
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlencode
from httpx._content import encode_json as httpx_encode_json
from httpx._content import encode_multipart_data, encode_urlencoded_data
from litestar import delete, patch, post, put
from litestar.app import Litestar
from litestar.connection import Request
from litestar.enums import HttpMethod, ParamType, RequestEncodingType, ScopeType
from litestar.handlers.http_handlers import get
from litestar.serialization import decode_json, default_serializer, encode_json
from litestar.types import DataContainerType, HTTPScope, RouteHandlerType
from litestar.types.asgi_types import ASGIVersion
from litestar.utils import get_serializer_from_scope
if TYPE_CHECKING:
    from httpx._types import FileTypes
    from litestar.datastructures.cookie import Cookie
    from litestar.handlers.http_handlers import HTTPRouteHandler
_decorator_http_method_map: dict[HttpMethod, type[HTTPRouteHandler]] = {HttpMethod.GET: get, HttpMethod.POST: post, HttpMethod.DELETE: delete, HttpMethod.PATCH: patch, HttpMethod.PUT: put}

def _create_default_route_handler(http_method: HttpMethod, handler_kwargs: dict[str, Any] | None, app: Litestar) -> HTTPRouteHandler:
    if False:
        for i in range(10):
            print('nop')
    handler_decorator = _decorator_http_method_map[http_method]

    def _default_route_handler() -> None:
        if False:
            print('Hello World!')
        ...
    handler = handler_decorator('/', sync_to_thread=False, **handler_kwargs or {})(_default_route_handler)
    handler.owner = app
    return handler

def _create_default_app() -> Litestar:
    if False:
        i = 10
        return i + 15
    return Litestar(route_handlers=[])

class RequestFactory:
    """Factory to create :class:`Request <litestar.connection.Request>` instances."""
    __slots__ = ('app', 'server', 'port', 'root_path', 'scheme', 'handler_kwargs', 'serializer')

    def __init__(self, app: Litestar | None=None, server: str='test.org', port: int=3000, root_path: str='', scheme: str='http', handler_kwargs: dict[str, Any] | None=None) -> None:
        if False:
            print('Hello World!')
        'Initialize ``RequestFactory``\n\n        Args:\n             app: An instance of :class:`Litestar <litestar.app.Litestar>` to set as ``request.scope["app"]``.\n             server: The server\'s domain.\n             port: The server\'s port.\n             root_path: Root path for the server.\n             scheme: Scheme for the server.\n             handler_kwargs: Kwargs to pass to the route handler created for the request\n\n        Examples:\n            .. code-block:: python\n\n                from litestar import Litestar\n                from litestar.enums import RequestEncodingType\n                from litestar.testing import RequestFactory\n\n                from tests import PersonFactory\n\n                my_app = Litestar(route_handlers=[])\n                my_server = "litestar.org"\n\n                # Create a GET request\n                query_params = {"id": 1}\n                get_user_request = RequestFactory(app=my_app, server=my_server).get(\n                    "/person", query_params=query_params\n                )\n\n                # Create a POST request\n                new_person = PersonFactory.build()\n                create_user_request = RequestFactory(app=my_app, server=my_server).post(\n                    "/person", data=person\n                )\n\n                # Create a request with a special header\n                headers = {"header1": "value1"}\n                request_with_header = RequestFactory(app=my_app, server=my_server).get(\n                    "/person", query_params=query_params, headers=headers\n                )\n\n                # Create a request with a media type\n                request_with_media_type = RequestFactory(app=my_app, server=my_server).post(\n                    "/person", data=person, request_media_type=RequestEncodingType.MULTI_PART\n                )\n\n        '
        self.app = app if app is not None else _create_default_app()
        self.server = server
        self.port = port
        self.root_path = root_path
        self.scheme = scheme
        self.handler_kwargs = handler_kwargs
        self.serializer = partial(default_serializer, type_encoders=self.app.type_encoders)

    def _create_scope(self, path: str, http_method: HttpMethod, session: dict[str, Any] | None=None, user: Any=None, auth: Any=None, query_params: dict[str, str | list[str]] | None=None, state: dict[str, Any] | None=None, path_params: dict[str, str] | None=None, http_version: str | None='1.1', route_handler: RouteHandlerType | None=None) -> HTTPScope:
        if False:
            while True:
                i = 10
        'Create the scope for the :class:`Request <litestar.connection.Request>`.\n\n        Args:\n            path: The request\'s path.\n            http_method: The request\'s HTTP method.\n            session: A dictionary of session data.\n            user: A value for `request.scope["user"]`.\n            auth: A value for `request.scope["auth"]`.\n            query_params: A dictionary of values from which the request\'s query will be generated.\n            state: Arbitrary request state.\n            path_params: A string keyed dictionary of path parameter values.\n            http_version: HTTP version. Defaults to "1.1".\n            route_handler: A route handler instance or method. If not provided a default handler is set.\n\n        Returns:\n            A dictionary that can be passed as a scope to the :class:`Request <litestar.connection.Request>` ctor.\n        '
        if session is None:
            session = {}
        if state is None:
            state = {}
        if path_params is None:
            path_params = {}
        return HTTPScope(type=ScopeType.HTTP, method=http_method.value, scheme=self.scheme, server=(self.server, self.port), root_path=self.root_path.rstrip('/'), path=path, headers=[], app=self.app, session=session, user=user, auth=auth, query_string=urlencode(query_params, doseq=True).encode() if query_params else b'', path_params=path_params, client=(self.server, self.port), state=state, asgi=ASGIVersion(spec_version='3.0', version='3.0'), http_version=http_version or '1.1', raw_path=path.encode('ascii'), route_handler=route_handler or _create_default_route_handler(http_method, self.handler_kwargs, app=self.app), extensions={})

    @classmethod
    def _create_cookie_header(cls, headers: dict[str, str], cookies: list[Cookie] | str | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Create the cookie header and add it to the ``headers`` dictionary.\n\n        Args:\n            headers: A dictionary of headers, the cookie header will be added to it.\n            cookies: A string representing the cookie header or a list of "Cookie" instances.\n                This value can include multiple cookies.\n        '
        if not cookies:
            return
        if isinstance(cookies, list):
            cookie_header = '; '.join((cookie.to_header(header='') for cookie in cookies))
            headers[ParamType.COOKIE] = cookie_header
        elif isinstance(cookies, str):
            headers[ParamType.COOKIE] = cookies

    def _build_headers(self, headers: dict[str, str] | None=None, cookies: list[Cookie] | str | None=None) -> list[tuple[bytes, bytes]]:
        if False:
            print('Hello World!')
        'Build a list of encoded headers that can be passed to the request scope.\n\n        Args:\n            headers: A dictionary of headers.\n            cookies: A string representing the cookie header or a list of "Cookie" instances.\n                This value can include multiple cookies.\n\n        Returns:\n            A list of encoded headers that can be passed to the request scope.\n        '
        headers = headers or {}
        self._create_cookie_header(headers, cookies)
        return [(key.lower().encode('latin-1', errors='ignore'), value.encode('latin-1', errors='ignore')) for (key, value) in headers.items()]

    def _create_request_with_data(self, http_method: HttpMethod, path: str, headers: dict[str, str] | None=None, cookies: list[Cookie] | str | None=None, session: dict[str, Any] | None=None, user: Any=None, auth: Any=None, request_media_type: RequestEncodingType=RequestEncodingType.JSON, data: dict[str, Any] | DataContainerType | None=None, files: dict[str, FileTypes] | list[tuple[str, FileTypes]] | None=None, query_params: dict[str, str | list[str]] | None=None, state: dict[str, Any] | None=None, path_params: dict[str, str] | None=None, http_version: str | None='1.1', route_handler: RouteHandlerType | None=None) -> Request[Any, Any, Any]:
        if False:
            while True:
                i = 10
        'Create a :class:`Request <litestar.connection.Request>` instance that has body (data)\n\n        Args:\n            http_method: The request\'s HTTP method.\n            path: The request\'s path.\n            headers: A dictionary of headers.\n            cookies: A string representing the cookie header or a list of "Cookie" instances.\n                This value can include multiple cookies.\n            session: A dictionary of session data.\n            user: A value for `request.scope["user"]`\n            auth: A value for `request.scope["auth"]`\n            request_media_type: The \'Content-Type\' header of the request.\n            data: A value for the request\'s body. Can be any supported serializable type.\n            files: A dictionary of files to be sent with the request.\n            query_params: A dictionary of values from which the request\'s query will be generated.\n            state: Arbitrary request state.\n            path_params: A string keyed dictionary of path parameter values.\n            http_version: HTTP version. Defaults to "1.1".\n            route_handler: A route handler instance or method. If not provided a default handler is set.\n\n        Returns:\n            A :class:`Request <litestar.connection.Request>` instance\n        '
        scope = self._create_scope(path=path, http_method=http_method, session=session, user=user, auth=auth, query_params=query_params, state=state, path_params=path_params, http_version=http_version, route_handler=route_handler)
        headers = headers or {}
        if data:
            data = json.loads(encode_json(data, serializer=get_serializer_from_scope(scope)))
            if request_media_type == RequestEncodingType.JSON:
                (encoding_headers, stream) = httpx_encode_json(data)
            elif request_media_type == RequestEncodingType.MULTI_PART:
                (encoding_headers, stream) = encode_multipart_data(cast('dict[str, Any]', data), files=files or [], boundary=None)
            else:
                (encoding_headers, stream) = encode_urlencoded_data(decode_json(value=encode_json(data)))
            headers.update(encoding_headers)
            body = b''
            for chunk in stream:
                body += chunk
            scope['_body'] = body
        else:
            scope['_body'] = b''
        self._create_cookie_header(headers, cookies)
        scope['headers'] = self._build_headers(headers)
        return Request(scope=scope)

    def get(self, path: str='/', headers: dict[str, str] | None=None, cookies: list[Cookie] | str | None=None, session: dict[str, Any] | None=None, user: Any=None, auth: Any=None, query_params: dict[str, str | list[str]] | None=None, state: dict[str, Any] | None=None, path_params: dict[str, str] | None=None, http_version: str | None='1.1', route_handler: RouteHandlerType | None=None) -> Request[Any, Any, Any]:
        if False:
            print('Hello World!')
        'Create a GET :class:`Request <litestar.connection.Request>` instance.\n\n        Args:\n            path: The request\'s path.\n            headers: A dictionary of headers.\n            cookies: A string representing the cookie header or a list of "Cookie" instances.\n                This value can include multiple cookies.\n            session: A dictionary of session data.\n            user: A value for `request.scope["user"]`.\n            auth: A value for `request.scope["auth"]`.\n            query_params: A dictionary of values from which the request\'s query will be generated.\n            state: Arbitrary request state.\n            path_params: A string keyed dictionary of path parameter values.\n            http_version: HTTP version. Defaults to "1.1".\n            route_handler: A route handler instance or method. If not provided a default handler is set.\n\n        Returns:\n            A :class:`Request <litestar.connection.Request>` instance\n        '
        scope = self._create_scope(path=path, http_method=HttpMethod.GET, session=session, user=user, auth=auth, query_params=query_params, state=state, path_params=path_params, http_version=http_version, route_handler=route_handler)
        scope['headers'] = self._build_headers(headers, cookies)
        return Request(scope=scope)

    def post(self, path: str='/', headers: dict[str, str] | None=None, cookies: list[Cookie] | str | None=None, session: dict[str, Any] | None=None, user: Any=None, auth: Any=None, request_media_type: RequestEncodingType=RequestEncodingType.JSON, data: dict[str, Any] | DataContainerType | None=None, query_params: dict[str, str | list[str]] | None=None, state: dict[str, Any] | None=None, path_params: dict[str, str] | None=None, http_version: str | None='1.1', route_handler: RouteHandlerType | None=None) -> Request[Any, Any, Any]:
        if False:
            i = 10
            return i + 15
        'Create a POST :class:`Request <litestar.connection.Request>` instance.\n\n        Args:\n            path: The request\'s path.\n            headers: A dictionary of headers.\n            cookies: A string representing the cookie header or a list of "Cookie" instances.\n                This value can include multiple cookies.\n            session: A dictionary of session data.\n            user: A value for `request.scope["user"]`.\n            auth: A value for `request.scope["auth"]`.\n            request_media_type: The \'Content-Type\' header of the request.\n            data: A value for the request\'s body. Can be any supported serializable type.\n            query_params: A dictionary of values from which the request\'s query will be generated.\n            state: Arbitrary request state.\n            path_params: A string keyed dictionary of path parameter values.\n            http_version: HTTP version. Defaults to "1.1".\n            route_handler: A route handler instance or method. If not provided a default handler is set.\n\n        Returns:\n            A :class:`Request <litestar.connection.Request>` instance\n        '
        return self._create_request_with_data(auth=auth, cookies=cookies, data=data, headers=headers, http_method=HttpMethod.POST, path=path, query_params=query_params, request_media_type=request_media_type, session=session, user=user, state=state, path_params=path_params, http_version=http_version, route_handler=route_handler)

    def put(self, path: str='/', headers: dict[str, str] | None=None, cookies: list[Cookie] | str | None=None, session: dict[str, Any] | None=None, user: Any=None, auth: Any=None, request_media_type: RequestEncodingType=RequestEncodingType.JSON, data: dict[str, Any] | DataContainerType | None=None, query_params: dict[str, str | list[str]] | None=None, state: dict[str, Any] | None=None, path_params: dict[str, str] | None=None, http_version: str | None='1.1', route_handler: RouteHandlerType | None=None) -> Request[Any, Any, Any]:
        if False:
            return 10
        'Create a PUT :class:`Request <litestar.connection.Request>` instance.\n\n        Args:\n            path: The request\'s path.\n            headers: A dictionary of headers.\n            cookies: A string representing the cookie header or a list of "Cookie" instances.\n                This value can include multiple cookies.\n            session: A dictionary of session data.\n            user: A value for `request.scope["user"]`.\n            auth: A value for `request.scope["auth"]`.\n            request_media_type: The \'Content-Type\' header of the request.\n            data: A value for the request\'s body. Can be any supported serializable type.\n            query_params: A dictionary of values from which the request\'s query will be generated.\n            state: Arbitrary request state.\n            path_params: A string keyed dictionary of path parameter values.\n            http_version: HTTP version. Defaults to "1.1".\n            route_handler: A route handler instance or method. If not provided a default handler is set.\n\n        Returns:\n            A :class:`Request <litestar.connection.Request>` instance\n        '
        return self._create_request_with_data(auth=auth, cookies=cookies, data=data, headers=headers, http_method=HttpMethod.PUT, path=path, query_params=query_params, request_media_type=request_media_type, session=session, user=user, state=state, path_params=path_params, http_version=http_version, route_handler=route_handler)

    def patch(self, path: str='/', headers: dict[str, str] | None=None, cookies: list[Cookie] | str | None=None, session: dict[str, Any] | None=None, user: Any=None, auth: Any=None, request_media_type: RequestEncodingType=RequestEncodingType.JSON, data: dict[str, Any] | DataContainerType | None=None, query_params: dict[str, str | list[str]] | None=None, state: dict[str, Any] | None=None, path_params: dict[str, str] | None=None, http_version: str | None='1.1', route_handler: RouteHandlerType | None=None) -> Request[Any, Any, Any]:
        if False:
            return 10
        'Create a PATCH :class:`Request <litestar.connection.Request>` instance.\n\n        Args:\n            path: The request\'s path.\n            headers: A dictionary of headers.\n            cookies: A string representing the cookie header or a list of "Cookie" instances.\n                This value can include multiple cookies.\n            session: A dictionary of session data.\n            user: A value for `request.scope["user"]`.\n            auth: A value for `request.scope["auth"]`.\n            request_media_type: The \'Content-Type\' header of the request.\n            data: A value for the request\'s body. Can be any supported serializable type.\n            query_params: A dictionary of values from which the request\'s query will be generated.\n            state: Arbitrary request state.\n            path_params: A string keyed dictionary of path parameter values.\n            http_version: HTTP version. Defaults to "1.1".\n            route_handler: A route handler instance or method. If not provided a default handler is set.\n\n        Returns:\n            A :class:`Request <litestar.connection.Request>` instance\n        '
        return self._create_request_with_data(auth=auth, cookies=cookies, data=data, headers=headers, http_method=HttpMethod.PATCH, path=path, query_params=query_params, request_media_type=request_media_type, session=session, user=user, state=state, path_params=path_params, http_version=http_version, route_handler=route_handler)

    def delete(self, path: str='/', headers: dict[str, str] | None=None, cookies: list[Cookie] | str | None=None, session: dict[str, Any] | None=None, user: Any=None, auth: Any=None, query_params: dict[str, str | list[str]] | None=None, state: dict[str, Any] | None=None, path_params: dict[str, str] | None=None, http_version: str | None='1.1', route_handler: RouteHandlerType | None=None) -> Request[Any, Any, Any]:
        if False:
            while True:
                i = 10
        'Create a POST :class:`Request <litestar.connection.Request>` instance.\n\n        Args:\n            path: The request\'s path.\n            headers: A dictionary of headers.\n            cookies: A string representing the cookie header or a list of "Cookie" instances.\n                This value can include multiple cookies.\n            session: A dictionary of session data.\n            user: A value for `request.scope["user"]`.\n            auth: A value for `request.scope["auth"]`.\n            query_params: A dictionary of values from which the request\'s query will be generated.\n            state: Arbitrary request state.\n            path_params: A string keyed dictionary of path parameter values.\n            http_version: HTTP version. Defaults to "1.1".\n            route_handler: A route handler instance or method. If not provided a default handler is set.\n\n        Returns:\n            A :class:`Request <litestar.connection.Request>` instance\n        '
        scope = self._create_scope(path=path, http_method=HttpMethod.DELETE, session=session, user=user, auth=auth, query_params=query_params, state=state, path_params=path_params, http_version=http_version, route_handler=route_handler)
        scope['headers'] = self._build_headers(headers, cookies)
        return Request(scope=scope)