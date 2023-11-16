from __future__ import annotations
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Generic, Mapping, Sequence, TypeVar
from urllib.parse import urljoin
from httpx import USE_CLIENT_DEFAULT, Client, Response
from litestar import HttpMethod
from litestar.testing.client.base import BaseTestClient
from litestar.testing.life_span_handler import LifeSpanHandler
from litestar.testing.transport import ConnectionUpgradeExceptionError, TestClientTransport
from litestar.types import AnyIOBackend, ASGIApp
if TYPE_CHECKING:
    from httpx._client import UseClientDefault
    from httpx._types import AuthTypes, CookieTypes, HeaderTypes, QueryParamTypes, RequestContent, RequestData, RequestFiles, TimeoutTypes, URLTypes
    from litestar.middleware.session.base import BaseBackendConfig
    from litestar.testing.websocket_test_session import WebSocketTestSession
T = TypeVar('T', bound=ASGIApp)

class TestClient(Client, BaseTestClient, Generic[T]):
    lifespan_handler: LifeSpanHandler[Any]
    exit_stack: ExitStack

    def __init__(self, app: T, base_url: str='http://testserver.local', raise_server_exceptions: bool=True, root_path: str='', backend: AnyIOBackend='asyncio', backend_options: Mapping[str, Any] | None=None, session_config: BaseBackendConfig | None=None, timeout: float | None=None, cookies: CookieTypes | None=None) -> None:
        if False:
            i = 10
            return i + 15
        'A client implementation providing a context manager for testing applications.\n\n        Args:\n            app: The instance of :class:`Litestar <litestar.app.Litestar>` under test.\n            base_url: URL scheme and domain for test request paths, e.g. \'http://testserver\'.\n            raise_server_exceptions: Flag for the underlying test client to raise server exceptions instead of\n                wrapping them in an HTTP response.\n            root_path: Path prefix for requests.\n            backend: The async backend to use, options are "asyncio" or "trio".\n            backend_options: ``anyio`` options.\n            session_config: Configuration for Session Middleware class to create raw session cookies for request to the\n                route handlers.\n            timeout: Request timeout\n            cookies: Cookies to set on the client.\n        '
        BaseTestClient.__init__(self, app=app, base_url=base_url, backend=backend, backend_options=backend_options, session_config=session_config, cookies=cookies)
        Client.__init__(self, app=self.app, base_url=base_url, headers={'user-agent': 'testclient'}, follow_redirects=True, cookies=cookies, transport=TestClientTransport(client=self, raise_server_exceptions=raise_server_exceptions, root_path=root_path), timeout=timeout)

    def __enter__(self) -> TestClient[T]:
        if False:
            print('Hello World!')
        with ExitStack() as stack:
            self.blocking_portal = portal = stack.enter_context(self.portal())
            self.lifespan_handler = LifeSpanHandler(client=self)

            @stack.callback
            def reset_portal() -> None:
                if False:
                    i = 10
                    return i + 15
                delattr(self, 'blocking_portal')

            @stack.callback
            def wait_shutdown() -> None:
                if False:
                    i = 10
                    return i + 15
                portal.call(self.lifespan_handler.wait_shutdown)
            self.exit_stack = stack.pop_all()
        return self

    def __exit__(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.exit_stack.close()

    def request(self, method: str, url: URLTypes, *, content: RequestContent | None=None, data: RequestData | None=None, files: RequestFiles | None=None, json: Any | None=None, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | UseClientDefault | None=USE_CLIENT_DEFAULT, follow_redirects: bool | UseClientDefault=USE_CLIENT_DEFAULT, timeout: TimeoutTypes | UseClientDefault=USE_CLIENT_DEFAULT, extensions: Mapping[str, Any] | None=None) -> Response:
        if False:
            return 10
        'Sends a request.\n\n        Args:\n            method: An HTTP method.\n            url: URL or path for the request.\n            content: Request content.\n            data: Form encoded data.\n            files: Multipart files to send.\n            json: JSON data to send.\n            params: Query parameters.\n            headers: Request headers.\n            cookies: Request cookies.\n            auth: Auth headers.\n            follow_redirects: Whether to follow redirects.\n            timeout: Request timeout.\n            extensions: Dictionary of ASGI extensions.\n\n        Returns:\n            An HTTPX Response.\n        '
        return Client.request(self, url=self.base_url.join(url), method=method.value if isinstance(method, HttpMethod) else method, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=None if extensions is None else dict(extensions))

    def get(self, url: URLTypes, *, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | UseClientDefault=USE_CLIENT_DEFAULT, follow_redirects: bool | UseClientDefault=USE_CLIENT_DEFAULT, timeout: TimeoutTypes | UseClientDefault=USE_CLIENT_DEFAULT, extensions: Mapping[str, Any] | None=None) -> Response:
        if False:
            i = 10
            return i + 15
        'Sends a GET request.\n\n        Args:\n            url: URL or path for the request.\n            params: Query parameters.\n            headers: Request headers.\n            cookies: Request cookies.\n            auth: Auth headers.\n            follow_redirects: Whether to follow redirects.\n            timeout: Request timeout.\n            extensions: Dictionary of ASGI extensions.\n\n        Returns:\n            An HTTPX Response.\n        '
        return Client.get(self, url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=None if extensions is None else dict(extensions))

    def options(self, url: URLTypes, *, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | UseClientDefault=USE_CLIENT_DEFAULT, follow_redirects: bool | UseClientDefault=USE_CLIENT_DEFAULT, timeout: TimeoutTypes | UseClientDefault=USE_CLIENT_DEFAULT, extensions: Mapping[str, Any] | None=None) -> Response:
        if False:
            return 10
        'Sends an OPTIONS request.\n\n        Args:\n            url: URL or path for the request.\n            params: Query parameters.\n            headers: Request headers.\n            cookies: Request cookies.\n            auth: Auth headers.\n            follow_redirects: Whether to follow redirects.\n            timeout: Request timeout.\n            extensions: Dictionary of ASGI extensions.\n\n        Returns:\n            An HTTPX Response.\n        '
        return Client.options(self, url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=None if extensions is None else dict(extensions))

    def head(self, url: URLTypes, *, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | UseClientDefault=USE_CLIENT_DEFAULT, follow_redirects: bool | UseClientDefault=USE_CLIENT_DEFAULT, timeout: TimeoutTypes | UseClientDefault=USE_CLIENT_DEFAULT, extensions: Mapping[str, Any] | None=None) -> Response:
        if False:
            for i in range(10):
                print('nop')
        'Sends a HEAD request.\n\n        Args:\n            url: URL or path for the request.\n            params: Query parameters.\n            headers: Request headers.\n            cookies: Request cookies.\n            auth: Auth headers.\n            follow_redirects: Whether to follow redirects.\n            timeout: Request timeout.\n            extensions: Dictionary of ASGI extensions.\n\n        Returns:\n            An HTTPX Response.\n        '
        return Client.head(self, url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=None if extensions is None else dict(extensions))

    def post(self, url: URLTypes, *, content: RequestContent | None=None, data: RequestData | None=None, files: RequestFiles | None=None, json: Any | None=None, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | UseClientDefault=USE_CLIENT_DEFAULT, follow_redirects: bool | UseClientDefault=USE_CLIENT_DEFAULT, timeout: TimeoutTypes | UseClientDefault=USE_CLIENT_DEFAULT, extensions: Mapping[str, Any] | None=None) -> Response:
        if False:
            return 10
        'Sends a POST request.\n\n        Args:\n            url: URL or path for the request.\n            content: Request content.\n            data: Form encoded data.\n            files: Multipart files to send.\n            json: JSON data to send.\n            params: Query parameters.\n            headers: Request headers.\n            cookies: Request cookies.\n            auth: Auth headers.\n            follow_redirects: Whether to follow redirects.\n            timeout: Request timeout.\n            extensions: Dictionary of ASGI extensions.\n\n        Returns:\n            An HTTPX Response.\n        '
        return Client.post(self, url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=None if extensions is None else dict(extensions))

    def put(self, url: URLTypes, *, content: RequestContent | None=None, data: RequestData | None=None, files: RequestFiles | None=None, json: Any | None=None, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | UseClientDefault=USE_CLIENT_DEFAULT, follow_redirects: bool | UseClientDefault=USE_CLIENT_DEFAULT, timeout: TimeoutTypes | UseClientDefault=USE_CLIENT_DEFAULT, extensions: Mapping[str, Any] | None=None) -> Response:
        if False:
            return 10
        'Sends a PUT request.\n\n        Args:\n            url: URL or path for the request.\n            content: Request content.\n            data: Form encoded data.\n            files: Multipart files to send.\n            json: JSON data to send.\n            params: Query parameters.\n            headers: Request headers.\n            cookies: Request cookies.\n            auth: Auth headers.\n            follow_redirects: Whether to follow redirects.\n            timeout: Request timeout.\n            extensions: Dictionary of ASGI extensions.\n\n        Returns:\n            An HTTPX Response.\n        '
        return Client.put(self, url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=None if extensions is None else dict(extensions))

    def patch(self, url: URLTypes, *, content: RequestContent | None=None, data: RequestData | None=None, files: RequestFiles | None=None, json: Any | None=None, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | UseClientDefault=USE_CLIENT_DEFAULT, follow_redirects: bool | UseClientDefault=USE_CLIENT_DEFAULT, timeout: TimeoutTypes | UseClientDefault=USE_CLIENT_DEFAULT, extensions: Mapping[str, Any] | None=None) -> Response:
        if False:
            for i in range(10):
                print('nop')
        'Sends a PATCH request.\n\n        Args:\n            url: URL or path for the request.\n            content: Request content.\n            data: Form encoded data.\n            files: Multipart files to send.\n            json: JSON data to send.\n            params: Query parameters.\n            headers: Request headers.\n            cookies: Request cookies.\n            auth: Auth headers.\n            follow_redirects: Whether to follow redirects.\n            timeout: Request timeout.\n            extensions: Dictionary of ASGI extensions.\n\n        Returns:\n            An HTTPX Response.\n        '
        return Client.patch(self, url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=None if extensions is None else dict(extensions))

    def delete(self, url: URLTypes, *, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | UseClientDefault=USE_CLIENT_DEFAULT, follow_redirects: bool | UseClientDefault=USE_CLIENT_DEFAULT, timeout: TimeoutTypes | UseClientDefault=USE_CLIENT_DEFAULT, extensions: Mapping[str, Any] | None=None) -> Response:
        if False:
            while True:
                i = 10
        'Sends a DELETE request.\n\n        Args:\n            url: URL or path for the request.\n            params: Query parameters.\n            headers: Request headers.\n            cookies: Request cookies.\n            auth: Auth headers.\n            follow_redirects: Whether to follow redirects.\n            timeout: Request timeout.\n            extensions: Dictionary of ASGI extensions.\n\n        Returns:\n            An HTTPX Response.\n        '
        return Client.delete(self, url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=None if extensions is None else dict(extensions))

    def websocket_connect(self, url: str, subprotocols: Sequence[str] | None=None, params: QueryParamTypes | None=None, headers: HeaderTypes | None=None, cookies: CookieTypes | None=None, auth: AuthTypes | UseClientDefault=USE_CLIENT_DEFAULT, follow_redirects: bool | UseClientDefault=USE_CLIENT_DEFAULT, timeout: TimeoutTypes | UseClientDefault=USE_CLIENT_DEFAULT, extensions: Mapping[str, Any] | None=None) -> WebSocketTestSession:
        if False:
            for i in range(10):
                print('nop')
        'Sends a GET request to establish a websocket connection.\n\n        Args:\n            url: Request URL.\n            subprotocols: Websocket subprotocols.\n            params: Query parameters.\n            headers: Request headers.\n            cookies: Request cookies.\n            auth: Auth headers.\n            follow_redirects: Whether to follow redirects.\n            timeout: Request timeout.\n            extensions: Dictionary of ASGI extensions.\n\n        Returns:\n            A `WebSocketTestSession <litestar.testing.WebSocketTestSession>` instance.\n        '
        url = urljoin('ws://testserver', url)
        default_headers: dict[str, str] = {}
        default_headers.setdefault('connection', 'upgrade')
        default_headers.setdefault('sec-websocket-key', 'testserver==')
        default_headers.setdefault('sec-websocket-version', '13')
        if subprotocols is not None:
            default_headers.setdefault('sec-websocket-protocol', ', '.join(subprotocols))
        try:
            Client.request(self, 'GET', url, headers={**dict(headers or {}), **default_headers}, params=params, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=None if extensions is None else dict(extensions))
        except ConnectionUpgradeExceptionError as exc:
            return exc.session
        raise RuntimeError('Expected WebSocket upgrade')

    def set_session_data(self, data: dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        'Set session data.\n\n        Args:\n            data: Session data\n\n        Returns:\n            None\n\n        Examples:\n            .. code-block:: python\n\n                from litestar import Litestar, get\n                from litestar.middleware.session.memory_backend import MemoryBackendConfig\n\n                session_config = MemoryBackendConfig()\n\n\n                @get(path="/test")\n                def get_session_data(request: Request) -> Dict[str, Any]:\n                    return request.session\n\n\n                app = Litestar(\n                    route_handlers=[get_session_data], middleware=[session_config.middleware]\n                )\n\n                with TestClient(app=app, session_config=session_config) as client:\n                    client.set_session_data({"foo": "bar"})\n                    assert client.get("/test").json() == {"foo": "bar"}\n\n        '
        with self.portal() as portal:
            portal.call(self._set_session_data, data)

    def get_session_data(self) -> dict[str, Any]:
        if False:
            print('Hello World!')
        'Get session data.\n\n        Returns:\n            A dictionary containing session data.\n\n        Examples:\n            .. code-block:: python\n\n                from litestar import Litestar, post\n                from litestar.middleware.session.memory_backend import MemoryBackendConfig\n\n                session_config = MemoryBackendConfig()\n\n\n                @post(path="/test")\n                def set_session_data(request: Request) -> None:\n                    request.session["foo"] == "bar"\n\n\n                app = Litestar(\n                    route_handlers=[set_session_data], middleware=[session_config.middleware]\n                )\n\n                with TestClient(app=app, session_config=session_config) as client:\n                    client.post("/test")\n                    assert client.get_session_data() == {"foo": "bar"}\n\n        '
        with self.portal() as portal:
            return portal.call(self._get_session_data)