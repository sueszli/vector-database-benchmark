import http.server
import threading
import logging
import pytest
from qutebrowser.qt.core import QUrl
from qutebrowser.qt.network import QNetworkProxy, QNetworkProxyQuery, QHostInfo, QHostAddress
from qutebrowser.browser.network import pac
pytestmark = pytest.mark.usefixtures('qapp')

def _pac_common_test(test_str):
    if False:
        for i in range(10):
            print('nop')
    fun_str_f = '\n        function FindProxyForURL(domain, host) {{\n            {}\n            return "DIRECT; PROXY 127.0.0.1:8080; SOCKS 192.168.1.1:4444";\n        }}\n    '
    fun_str = fun_str_f.format(test_str)
    res = pac.PACResolver(fun_str)
    proxies = res.resolve(QNetworkProxyQuery(QUrl('https://example.com/test')))
    assert len(proxies) == 3
    assert proxies[0].type() == QNetworkProxy.ProxyType.NoProxy
    assert proxies[1].type() == QNetworkProxy.ProxyType.HttpProxy
    assert proxies[1].hostName() == '127.0.0.1'
    assert proxies[1].port() == 8080
    assert proxies[2].type() == QNetworkProxy.ProxyType.Socks5Proxy
    assert proxies[2].hostName() == '192.168.1.1'
    assert proxies[2].port() == 4444

def _pac_equality_test(call, expected):
    if False:
        while True:
            i = 10
    test_str_f = '\n        var res = ({0});\n        var expected = ({1});\n        if(res !== expected) {{\n            throw new Error("failed test {0}: got \'" + res + "\', expected \'" + expected + "\'");\n        }}\n    '
    _pac_common_test(test_str_f.format(call, expected))

def _pac_except_test(caplog, call):
    if False:
        while True:
            i = 10
    test_str_f = '\n        var thrown = false;\n        try {{\n            var res = ({0});\n        }} catch(e) {{\n            thrown = true;\n        }}\n        if(!thrown) {{\n            throw new Error("failed test {0}: got \'" + res + "\', expected exception");\n        }}\n    '
    with caplog.at_level(logging.ERROR):
        _pac_common_test(test_str_f.format(call))

def _pac_noexcept_test(call):
    if False:
        i = 10
        return i + 15
    test_str_f = '\n        var res = ({0});\n    '
    _pac_common_test(test_str_f.format(call))

@pytest.mark.parametrize('domain, expected', [('known.domain', "'1.2.3.4'"), ('bogus.domain.foobar', 'null')])
def test_dnsResolve(monkeypatch, domain, expected):
    if False:
        i = 10
        return i + 15

    def mock_fromName(host):
        if False:
            print('Hello World!')
        info = QHostInfo()
        if host == 'known.domain':
            info.setAddresses([QHostAddress('1.2.3.4')])
        return info
    monkeypatch.setattr(QHostInfo, 'fromName', mock_fromName)
    _pac_equality_test("dnsResolve('{}')".format(domain), expected)

def test_myIpAddress():
    if False:
        while True:
            i = 10
    _pac_equality_test('isResolvable(myIpAddress())', 'true')

@pytest.mark.parametrize('host, expected', [('example', 'true'), ('example.com', 'false'), ('www.example.com', 'false')])
def test_isPlainHostName(host, expected):
    if False:
        return 10
    _pac_equality_test("isPlainHostName('{}')".format(host), expected)

def test_proxyBindings():
    if False:
        while True:
            i = 10
    _pac_equality_test('JSON.stringify(ProxyConfig.bindings)', "'{}'")

def test_invalid_port():
    if False:
        while True:
            i = 10
    test_str = '\n        function FindProxyForURL(domain, host) {\n            return "PROXY 127.0.0.1:FOO";\n        }\n    '
    res = pac.PACResolver(test_str)
    with pytest.raises(pac.ParseProxyError):
        res.resolve(QNetworkProxyQuery(QUrl('https://example.com/test')))

@pytest.mark.parametrize('string', ['', '{'])
def test_wrong_pac_string(string):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(pac.EvalProxyError):
        pac.PACResolver(string)

@pytest.mark.parametrize('value', ['', 'DIRECT FOO', 'PROXY', 'SOCKS', 'FOOBAR'])
def test_fail_parse(value):
    if False:
        i = 10
        return i + 15
    test_str_f = '\n        function FindProxyForURL(domain, host) {{\n            return "{}";\n        }}\n    '
    res = pac.PACResolver(test_str_f.format(value))
    with pytest.raises(pac.ParseProxyError):
        res.resolve(QNetworkProxyQuery(QUrl('https://example.com/test')))

def test_fail_return():
    if False:
        print('Hello World!')
    test_str = '\n        function FindProxyForURL(domain, host) {\n            return null;\n        }\n    '
    res = pac.PACResolver(test_str)
    with pytest.raises(pac.EvalProxyError):
        res.resolve(QNetworkProxyQuery(QUrl('https://example.com/test')))

@pytest.mark.parametrize('url, has_secret', [('http://example.com/secret', True), ('http://example.com?secret=yes', True), ('http://secret@example.com', False), ('http://user:secret@example.com', False), ('https://example.com/secret', False), ('https://example.com?secret=yes', False), ('https://secret@example.com', False), ('https://user:secret@example.com', False)])
@pytest.mark.parametrize('from_file', [True, False])
def test_secret_url(url, has_secret, from_file):
    if False:
        return 10
    'Make sure secret parts in a URL are stripped correctly.\n\n    The following parts are considered secret:\n        - If the PAC info is loaded from a local file, nothing.\n        - If the URL to resolve is a HTTP URL, the username/password.\n        - If the URL to resolve is a HTTPS URL, the username/password, query\n          and path.\n    '
    test_str = '\n        function FindProxyForURL(domain, host) {{\n            has_secret = domain.indexOf("secret") !== -1;\n            expected_secret = {};\n            if (has_secret !== expected_secret) {{\n                throw new Error("Expected secret: " + expected_secret + ", found: " + has_secret + " in " + domain);\n            }}\n            return "DIRECT";\n        }}\n    '.format('true' if has_secret or from_file else 'false')
    res = pac.PACResolver(test_str)
    res.resolve(QNetworkProxyQuery(QUrl(url)), from_file=from_file)

def test_logging(qtlog):
    if False:
        for i in range(10):
            print('nop')
    'Make sure console.log() works for PAC files.'
    test_str = '\n        function FindProxyForURL(domain, host) {\n            console.log("logging test");\n            return "DIRECT";\n        }\n    '
    res = pac.PACResolver(test_str)
    res.resolve(QNetworkProxyQuery(QUrl('https://example.com/test')))
    assert len(qtlog.records) == 1
    assert qtlog.records[0].message == 'logging test'

def fetcher_test(test_str):
    if False:
        print('Hello World!')

    class PACHandler(http.server.BaseHTTPRequestHandler):

        def do_GET(self):
            if False:
                for i in range(10):
                    print('nop')
            self.send_response(200)
            self.send_header('Content-type', 'application/x-ns-proxy-autoconfig')
            self.end_headers()
            self.wfile.write(test_str.encode('ascii'))
    ready_event = threading.Event()

    def serve():
        if False:
            return 10
        httpd = http.server.HTTPServer(('127.0.0.1', 8081), PACHandler)
        ready_event.set()
        httpd.handle_request()
        httpd.server_close()
    serve_thread = threading.Thread(target=serve, daemon=True)
    serve_thread.start()
    try:
        ready_event.wait()
        fetcher = pac.PACFetcher(QUrl('pac+http://127.0.0.1:8081'))
        fetcher.fetch()
        assert fetcher.fetch_error() is None
    finally:
        serve_thread.join()
    return fetcher

def test_fetch_success():
    if False:
        for i in range(10):
            print('nop')
    test_str = '\n        function FindProxyForURL(domain, host) {\n            return "DIRECT; PROXY 127.0.0.1:8080; SOCKS 192.168.1.1:4444";\n        }\n    '
    res = fetcher_test(test_str)
    proxies = res.resolve(QNetworkProxyQuery(QUrl('https://example.com/test')))
    assert len(proxies) == 3

def test_fetch_evalerror(caplog):
    if False:
        while True:
            i = 10
    test_str = '\n        function FindProxyForURL(domain, host) {\n            return "FOO";\n        }\n    '
    res = fetcher_test(test_str)
    with caplog.at_level(logging.ERROR):
        proxies = res.resolve(QNetworkProxyQuery(QUrl('https://example.com/test')))
    assert len(proxies) == 1
    assert proxies[0].port() == 9