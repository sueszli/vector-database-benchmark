from typing import TYPE_CHECKING, Any, cast
import pytest
from litestar import get
from litestar.config.allowed_hosts import AllowedHostsConfig
from litestar.exceptions import ImproperlyConfiguredException
from litestar.middleware import MiddlewareProtocol
from litestar.middleware.allowed_hosts import AllowedHostsMiddleware
from litestar.status_codes import HTTP_200_OK, HTTP_400_BAD_REQUEST
from litestar.testing import create_test_client
if TYPE_CHECKING:
    from litestar.types import Receive, Scope, Send

class DummyApp(MiddlewareProtocol):

    async def __call__(self, scope: 'Scope', receive: 'Receive', send: 'Send') -> None:
        return

def test_allowed_hosts_middleware() -> None:
    if False:
        while True:
            i = 10

    @get(path='/')
    def handler() -> None:
        if False:
            return 10
        ...
    client = create_test_client(route_handlers=[handler], allowed_hosts=['*.example.com', 'moishe.zuchmir.com'])
    unpacked_middleware = []
    cur = client.app.asgi_router.root_route_map_node.children['/'].asgi_handlers['GET'][0]
    while hasattr(cur, 'app'):
        unpacked_middleware.append(cur)
        cur = cast('Any', cur.app)
    unpacked_middleware.append(cur)
    assert len(unpacked_middleware) == 4
    allowed_hosts_middleware = cast('Any', unpacked_middleware[1])
    assert isinstance(allowed_hosts_middleware, AllowedHostsMiddleware)
    assert allowed_hosts_middleware.allowed_hosts_regex.pattern == '.*\\.example.com$|moishe.zuchmir.com'

def test_allowed_hosts_middleware_hosts_regex() -> None:
    if False:
        for i in range(10):
            print('nop')
    config = AllowedHostsConfig(allowed_hosts=['*.example.com', 'moishe.zuchmir.com'])
    middleware = AllowedHostsMiddleware(app=DummyApp(), config=config)
    assert middleware.allowed_hosts_regex is not None
    assert middleware.allowed_hosts_regex.pattern == '.*\\.example.com$|moishe.zuchmir.com'
    assert middleware.allowed_hosts_regex.fullmatch('www.example.com')
    assert middleware.allowed_hosts_regex.fullmatch('other.example.com')
    assert middleware.allowed_hosts_regex.fullmatch('x.y.z.example.com')
    assert middleware.allowed_hosts_regex.fullmatch('moishe.zuchmir.com')
    assert not middleware.allowed_hosts_regex.fullmatch('www.example.x.com')
    assert not middleware.allowed_hosts_regex.fullmatch('josh.zuchmir.com')
    assert not middleware.allowed_hosts_regex.fullmatch('x.moishe.zuchmir.com')
    assert not middleware.allowed_hosts_regex.fullmatch('moishe.zuchmir.x.com')

def test_allowed_hosts_middleware_redirect_regex() -> None:
    if False:
        while True:
            i = 10
    config = AllowedHostsConfig(allowed_hosts=['*.example.com', 'www.moishe.zuchmir.com', 'www.yada.bada.bing.io', 'example.com'])
    middleware = AllowedHostsMiddleware(app=DummyApp(), config=config)
    assert middleware.redirect_domains is not None
    assert middleware.redirect_domains.pattern == 'moishe.zuchmir.com|yada.bada.bing.io'
    assert middleware.redirect_domains.fullmatch('moishe.zuchmir.com')
    assert middleware.redirect_domains.fullmatch('yada.bada.bing.io')

def test_middleware_allowed_hosts() -> None:
    if False:
        return 10

    @get('/')
    def handler() -> dict:
        if False:
            while True:
                i = 10
        return {'hello': 'world'}
    config = AllowedHostsConfig(allowed_hosts=['*.example.com', 'moishe.zuchmir.com'])
    with create_test_client(handler, allowed_hosts=config) as client:
        client.base_url = 'http://x.example.com'
        response = client.get('/')
        assert response.status_code == HTTP_200_OK
        client.base_url = 'http://x.y.example.com'
        response = client.get('/')
        assert response.status_code == HTTP_200_OK
        client.base_url = 'http://moishe.zuchmir.com'
        response = client.get('/')
        assert response.status_code == HTTP_200_OK
        client.base_url = 'http://x.moishe.zuchmir.com'
        response = client.get('/')
        assert response.status_code == HTTP_400_BAD_REQUEST
        client.base_url = 'http://x.example.x.com'
        response = client.get('/')
        assert response.status_code == HTTP_400_BAD_REQUEST

def test_middleware_allow_all() -> None:
    if False:
        i = 10
        return i + 15

    @get('/')
    def handler() -> dict:
        if False:
            while True:
                i = 10
        return {'hello': 'world'}
    config = AllowedHostsConfig(allowed_hosts=['*', '*.example.com', 'moishe.zuchmir.com'])
    with create_test_client(handler, allowed_hosts=config) as client:
        client.base_url = 'http://any.domain.allowed.com'
        response = client.get('/')
        assert response.status_code == HTTP_200_OK

def test_middleware_redirect_on_www_by_default() -> None:
    if False:
        while True:
            i = 10

    @get('/')
    def handler() -> dict:
        if False:
            i = 10
            return i + 15
        return {'hello': 'world'}
    config = AllowedHostsConfig(allowed_hosts=['www.moishe.zuchmir.com'])
    with create_test_client(handler, allowed_hosts=config) as client:
        client.base_url = 'http://moishe.zuchmir.com'
        response = client.get('/')
        assert response.status_code == HTTP_200_OK
        assert str(response.url) == 'http://www.moishe.zuchmir.com/'

def test_middleware_does_not_redirect_when_off() -> None:
    if False:
        return 10

    @get('/')
    def handler() -> dict:
        if False:
            return 10
        return {'hello': 'world'}
    config = AllowedHostsConfig(allowed_hosts=['www.moishe.zuchmir.com'], www_redirect=False)
    with create_test_client(handler, allowed_hosts=config) as client:
        client.base_url = 'http://moishe.zuchmir.com'
        response = client.get('/')
        assert response.status_code == HTTP_400_BAD_REQUEST

def test_validation_raises_for_wrong_wildcard_domain() -> None:
    if False:
        while True:
            i = 10
    with pytest.raises(ImproperlyConfiguredException):
        AllowedHostsConfig(allowed_hosts=['www.moishe.*.com'])