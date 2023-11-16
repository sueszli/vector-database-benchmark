import httpcore
import pytest
import httpx

def url_to_origin(url: str) -> httpcore.URL:
    if False:
        while True:
            i = 10
    "\n    Given a URL string, return the origin in the raw tuple format that\n    `httpcore` uses for it's representation.\n    "
    u = httpx.URL(url)
    return httpcore.URL(scheme=u.raw_scheme, host=u.raw_host, port=u.port, target='/')

@pytest.mark.parametrize(['proxies', 'expected_proxies'], [('http://127.0.0.1', [('all://', 'http://127.0.0.1')]), ({'all://': 'http://127.0.0.1'}, [('all://', 'http://127.0.0.1')]), ({'http://': 'http://127.0.0.1', 'https://': 'https://127.0.0.1'}, [('http://', 'http://127.0.0.1'), ('https://', 'https://127.0.0.1')]), (httpx.Proxy('http://127.0.0.1'), [('all://', 'http://127.0.0.1')]), ({'https://': httpx.Proxy('https://127.0.0.1'), 'all://': 'http://127.0.0.1'}, [('all://', 'http://127.0.0.1'), ('https://', 'https://127.0.0.1')])])
def test_proxies_parameter(proxies, expected_proxies):
    if False:
        return 10
    client = httpx.Client(proxies=proxies)
    client_patterns = [p.pattern for p in client._mounts.keys()]
    client_proxies = list(client._mounts.values())
    for (proxy_key, url) in expected_proxies:
        assert proxy_key in client_patterns
        proxy = client_proxies[client_patterns.index(proxy_key)]
        assert isinstance(proxy, httpx.HTTPTransport)
        assert isinstance(proxy._pool, httpcore.HTTPProxy)
        assert proxy._pool._proxy_url == url_to_origin(url)
    assert len(expected_proxies) == len(client._mounts)

def test_socks_proxy():
    if False:
        print('Hello World!')
    url = httpx.URL('http://www.example.com')
    client = httpx.Client(proxies='socks5://localhost/')
    transport = client._transport_for_url(url)
    assert isinstance(transport, httpx.HTTPTransport)
    assert isinstance(transport._pool, httpcore.SOCKSProxy)
    async_client = httpx.AsyncClient(proxies='socks5://localhost/')
    async_transport = async_client._transport_for_url(url)
    assert isinstance(async_transport, httpx.AsyncHTTPTransport)
    assert isinstance(async_transport._pool, httpcore.AsyncSOCKSProxy)
PROXY_URL = 'http://[::1]'

@pytest.mark.parametrize(['url', 'proxies', 'expected'], [('http://example.com', None, None), ('http://example.com', {}, None), ('http://example.com', {'https://': PROXY_URL}, None), ('http://example.com', {'http://example.net': PROXY_URL}, None), ('http://example.com', {'http://*': PROXY_URL}, PROXY_URL), ('https://example.com', {'http://*': PROXY_URL}, None), ('http://example.com', {'http://example.com': PROXY_URL}, PROXY_URL), ('http://www.example.com', {'http://example.com': PROXY_URL}, None), ('http://example.com', {'http://*.example.com': PROXY_URL}, None), ('http://www.example.com', {'http://*.example.com': PROXY_URL}, PROXY_URL), ('http://example.com', {'http://*example.com': PROXY_URL}, PROXY_URL), ('http://www.example.com', {'http://*example.com': PROXY_URL}, PROXY_URL), ('http://wwwexample.com', {'http://*example.com': PROXY_URL}, None), ('http://example.com:443', {'http://example.com': PROXY_URL}, PROXY_URL), ('http://example.com', {'all://': PROXY_URL}, PROXY_URL), ('http://example.com', {'all://': PROXY_URL, 'http://example.com': None}, None), ('http://example.com', {'http://': PROXY_URL}, PROXY_URL), ('http://example.com', {'all://example.com': PROXY_URL}, PROXY_URL), ('http://example.com', {'http://example.com': PROXY_URL}, PROXY_URL), ('http://example.com', {'http://example.com:80': PROXY_URL}, PROXY_URL), ('http://example.com:8080', {'http://example.com:8080': PROXY_URL}, PROXY_URL), ('http://example.com:8080', {'http://example.com': PROXY_URL}, PROXY_URL), ('http://example.com', {'all://': PROXY_URL + ':1', 'http://': PROXY_URL + ':2', 'all://example.com': PROXY_URL + ':3', 'http://example.com': PROXY_URL + ':4'}, PROXY_URL + ':4'), ('http://example.com', {'all://': PROXY_URL + ':1', 'http://': PROXY_URL + ':2', 'all://example.com': PROXY_URL + ':3'}, PROXY_URL + ':3'), ('http://example.com', {'all://': PROXY_URL + ':1', 'http://': PROXY_URL + ':2'}, PROXY_URL + ':2')])
def test_transport_for_request(url, proxies, expected):
    if False:
        for i in range(10):
            print('nop')
    client = httpx.Client(proxies=proxies)
    transport = client._transport_for_url(httpx.URL(url))
    if expected is None:
        assert transport is client._transport
    else:
        assert isinstance(transport, httpx.HTTPTransport)
        assert isinstance(transport._pool, httpcore.HTTPProxy)
        assert transport._pool._proxy_url == url_to_origin(expected)

@pytest.mark.anyio
@pytest.mark.network
async def test_async_proxy_close():
    try:
        client = httpx.AsyncClient(proxies={'https://': PROXY_URL})
        await client.get('http://example.com')
    finally:
        await client.aclose()

@pytest.mark.network
def test_sync_proxy_close():
    if False:
        while True:
            i = 10
    try:
        client = httpx.Client(proxies={'https://': PROXY_URL})
        client.get('http://example.com')
    finally:
        client.close()

def test_unsupported_proxy_scheme():
    if False:
        return 10
    with pytest.raises(ValueError):
        httpx.Client(proxies='ftp://127.0.0.1')

@pytest.mark.parametrize(['url', 'env', 'expected'], [('http://google.com', {}, None), ('http://google.com', {'HTTP_PROXY': 'http://example.com'}, 'http://example.com'), ('http://google.com', {'HTTP_PROXY': 'example.com'}, 'http://example.com'), ('http://google.com', {'HTTP_PROXY': 'http://example.com', 'NO_PROXY': 'google.com'}, None), ('http://127.0.0.1', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': ''}, 'http://localhost:123'), ('http://127.0.0.1', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': '127.0.0.1'}, None), ('http://127.0.0.1', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'https://127.0.0.1'}, 'http://localhost:123'), ('http://127.0.0.1', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': '1.1.1.1'}, 'http://localhost:123'), ('http://courses.mit.edu', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'mit.edu'}, None), ('https://mit.edu.info', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'mit.edu'}, 'http://localhost:123'), ('https://mit.edu.info', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'mit.edu,edu.info'}, None), ('https://mit.edu.info', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'mit.edu, edu.info'}, None), ('https://mit.edu.info', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'mit.edu,mit.info'}, 'http://localhost:123'), ('https://foo.example.com', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'www.example.com'}, 'http://localhost:123'), ('https://www.example1.com', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': '.example1.com'}, None), ('https://www.example2.com', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': 'ample2.com'}, 'http://localhost:123'), ('https://www.example3.com', {'ALL_PROXY': 'http://localhost:123', 'NO_PROXY': '*'}, None)])
@pytest.mark.parametrize('client_class', [httpx.Client, httpx.AsyncClient])
def test_proxies_environ(monkeypatch, client_class, url, env, expected):
    if False:
        print('Hello World!')
    for (name, value) in env.items():
        monkeypatch.setenv(name, value)
    client = client_class()
    transport = client._transport_for_url(httpx.URL(url))
    if expected is None:
        assert transport == client._transport
    else:
        assert transport._pool._proxy_url == url_to_origin(expected)

@pytest.mark.parametrize(['proxies', 'is_valid'], [({'http': 'http://127.0.0.1'}, False), ({'https': 'http://127.0.0.1'}, False), ({'all': 'http://127.0.0.1'}, False), ({'http://': 'http://127.0.0.1'}, True), ({'https://': 'http://127.0.0.1'}, True), ({'all://': 'http://127.0.0.1'}, True)])
def test_for_deprecated_proxy_params(proxies, is_valid):
    if False:
        while True:
            i = 10
    if not is_valid:
        with pytest.raises(ValueError):
            httpx.Client(proxies=proxies)
    else:
        httpx.Client(proxies=proxies)