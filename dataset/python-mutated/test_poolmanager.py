from __future__ import annotations
import gzip
import typing
from test import LONG_TIMEOUT
from unittest import mock
import pytest
from dummyserver.testcase import HTTPDummyServerTestCase, IPv6HTTPDummyServerTestCase
from dummyserver.tornadoserver import HAS_IPV6
from urllib3 import HTTPHeaderDict, HTTPResponse, request
from urllib3.connectionpool import port_by_scheme
from urllib3.exceptions import MaxRetryError, URLSchemeUnknown
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry

class TestPoolManager(HTTPDummyServerTestCase):

    @classmethod
    def setup_class(cls) -> None:
        if False:
            return 10
        super().setup_class()
        cls.base_url = f'http://{cls.host}:{cls.port}'
        cls.base_url_alt = f'http://{cls.host_alt}:{cls.port}'

    def test_redirect(self) -> None:
        if False:
            print('Hello World!')
        with PoolManager() as http:
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url}/'}, redirect=False)
            assert r.status == 303
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url}/'})
            assert r.status == 200
            assert r.data == b'Dummy server!'

    def test_redirect_twice(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with PoolManager() as http:
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url}/redirect'}, redirect=False)
            assert r.status == 303
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url}/redirect?target={self.base_url}/'})
            assert r.status == 200
            assert r.data == b'Dummy server!'

    def test_redirect_to_relative_url(self) -> None:
        if False:
            return 10
        with PoolManager() as http:
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': '/redirect'}, redirect=False)
            assert r.status == 303
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': '/redirect'})
            assert r.status == 200
            assert r.data == b'Dummy server!'

    def test_cross_host_redirect(self) -> None:
        if False:
            i = 10
            return i + 15
        with PoolManager() as http:
            cross_host_location = f'{self.base_url_alt}/echo?a=b'
            with pytest.raises(MaxRetryError):
                http.request('GET', f'{self.base_url}/redirect', fields={'target': cross_host_location}, timeout=LONG_TIMEOUT, retries=0)
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url_alt}/echo?a=b'}, timeout=LONG_TIMEOUT, retries=1)
            assert isinstance(r, HTTPResponse)
            assert r._pool is not None
            assert r._pool.host == self.host_alt

    def test_too_many_redirects(self) -> None:
        if False:
            while True:
                i = 10
        with PoolManager() as http:
            with pytest.raises(MaxRetryError):
                http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url}/redirect?target={self.base_url}/'}, retries=1, preload_content=False)
            with pytest.raises(MaxRetryError):
                http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url}/redirect?target={self.base_url}/'}, retries=Retry(total=None, redirect=1), preload_content=False)
            assert len(http.pools) == 1
            pool = http.connection_from_host(self.host, self.port)
            assert pool.num_connections == 1

    def test_redirect_cross_host_remove_headers(self) -> None:
        if False:
            return 10
        with PoolManager() as http:
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url_alt}/headers'}, headers={'Authorization': 'foo', 'Cookie': 'foo=bar'})
            assert r.status == 200
            data = r.json()
            assert 'Authorization' not in data
            assert 'Cookie' not in data
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url_alt}/headers'}, headers={'authorization': 'foo', 'cookie': 'foo=bar'})
            assert r.status == 200
            data = r.json()
            assert 'authorization' not in data
            assert 'Authorization' not in data
            assert 'cookie' not in data
            assert 'Cookie' not in data

    def test_redirect_cross_host_no_remove_headers(self) -> None:
        if False:
            i = 10
            return i + 15
        with PoolManager() as http:
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url_alt}/headers'}, headers={'Authorization': 'foo', 'Cookie': 'foo=bar'}, retries=Retry(remove_headers_on_redirect=[]))
            assert r.status == 200
            data = r.json()
            assert data['Authorization'] == 'foo'
            assert data['Cookie'] == 'foo=bar'

    def test_redirect_cross_host_set_removed_headers(self) -> None:
        if False:
            while True:
                i = 10
        with PoolManager() as http:
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url_alt}/headers'}, headers={'X-API-Secret': 'foo', 'Authorization': 'bar', 'Cookie': 'foo=bar'}, retries=Retry(remove_headers_on_redirect=['X-API-Secret']))
            assert r.status == 200
            data = r.json()
            assert 'X-API-Secret' not in data
            assert data['Authorization'] == 'bar'
            assert data['Cookie'] == 'foo=bar'
            headers = {'x-api-secret': 'foo', 'authorization': 'bar', 'cookie': 'foo=bar'}
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url_alt}/headers'}, headers=headers, retries=Retry(remove_headers_on_redirect=['X-API-Secret']))
            assert r.status == 200
            data = r.json()
            assert 'x-api-secret' not in data
            assert 'X-API-Secret' not in data
            assert data['Authorization'] == 'bar'
            assert data['Cookie'] == 'foo=bar'
            assert headers == {'x-api-secret': 'foo', 'authorization': 'bar', 'cookie': 'foo=bar'}

    def test_redirect_without_preload_releases_connection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with PoolManager(block=True, maxsize=2) as http:
            r = http.request('GET', f'{self.base_url}/redirect', preload_content=False)
            assert isinstance(r, HTTPResponse)
            assert r._pool is not None
            assert r._pool.num_requests == 2
            assert r._pool.num_connections == 1
            assert len(http.pools) == 1

    def test_303_redirect_makes_request_lose_body(self) -> None:
        if False:
            while True:
                i = 10
        with PoolManager() as http:
            response = http.request('POST', f'{self.base_url}/redirect', fields={'target': f'{self.base_url}/headers_and_params', 'status': '303 See Other'})
        data = response.json()
        assert data['params'] == {}
        assert 'Content-Type' not in HTTPHeaderDict(data['headers'])

    def test_unknown_scheme(self) -> None:
        if False:
            while True:
                i = 10
        with PoolManager() as http:
            unknown_scheme = 'unknown'
            unknown_scheme_url = f'{unknown_scheme}://host'
            with pytest.raises(URLSchemeUnknown) as e:
                r = http.request('GET', unknown_scheme_url)
            assert e.value.scheme == unknown_scheme
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': unknown_scheme_url}, redirect=False)
            assert r.status == 303
            assert r.headers.get('Location') == unknown_scheme_url
            with pytest.raises(URLSchemeUnknown) as e:
                r = http.request('GET', f'{self.base_url}/redirect', fields={'target': unknown_scheme_url})
            assert e.value.scheme == unknown_scheme

    def test_raise_on_redirect(self) -> None:
        if False:
            return 10
        with PoolManager() as http:
            r = http.request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url}/redirect?target={self.base_url}/'}, retries=Retry(total=None, redirect=1, raise_on_redirect=False))
            assert r.status == 303

    def test_raise_on_status(self) -> None:
        if False:
            while True:
                i = 10
        with PoolManager() as http:
            with pytest.raises(MaxRetryError):
                r = http.request('GET', f'{self.base_url}/status', fields={'status': '500 Internal Server Error'}, retries=Retry(total=1, status_forcelist=range(500, 600)))
            with pytest.raises(MaxRetryError):
                r = http.request('GET', f'{self.base_url}/status', fields={'status': '500 Internal Server Error'}, retries=Retry(total=1, status_forcelist=range(500, 600), raise_on_status=True))
            r = http.request('GET', f'{self.base_url}/status', fields={'status': '500 Internal Server Error'}, retries=Retry(total=1, status_forcelist=range(500, 600), raise_on_status=False))
            assert r.status == 500

    def test_missing_port(self) -> None:
        if False:
            print('Hello World!')
        with PoolManager() as http:
            port_by_scheme['http'] = self.port
            try:
                r = http.request('GET', f'http://{self.host}/', retries=0)
            finally:
                port_by_scheme['http'] = 80
            assert r.status == 200
            assert r.data == b'Dummy server!'

    def test_headers(self) -> None:
        if False:
            return 10
        with PoolManager(headers={'Foo': 'bar'}) as http:
            r = http.request('GET', f'{self.base_url}/headers')
            returned_headers = r.json()
            assert returned_headers.get('Foo') == 'bar'
            r = http.request('POST', f'{self.base_url}/headers')
            returned_headers = r.json()
            assert returned_headers.get('Foo') == 'bar'
            r = http.request_encode_url('GET', f'{self.base_url}/headers')
            returned_headers = r.json()
            assert returned_headers.get('Foo') == 'bar'
            r = http.request_encode_body('POST', f'{self.base_url}/headers')
            returned_headers = r.json()
            assert returned_headers.get('Foo') == 'bar'
            r = http.request_encode_url('GET', f'{self.base_url}/headers', headers={'Baz': 'quux'})
            returned_headers = r.json()
            assert returned_headers.get('Foo') is None
            assert returned_headers.get('Baz') == 'quux'
            r = http.request_encode_body('GET', f'{self.base_url}/headers', headers={'Baz': 'quux'})
            returned_headers = r.json()
            assert returned_headers.get('Foo') is None
            assert returned_headers.get('Baz') == 'quux'

    def test_headers_http_header_dict(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        headers = HTTPHeaderDict()
        headers.add('Foo', 'bar')
        headers.add('Multi', '1')
        headers.add('Baz', 'quux')
        headers.add('Multi', '2')
        with PoolManager(headers=headers) as http:
            r = http.request('GET', f'{self.base_url}/multi_headers')
            returned_headers = r.json()['headers']
            assert returned_headers[-4:] == [['Foo', 'bar'], ['Multi', '1'], ['Multi', '2'], ['Baz', 'quux']]
            r = http.request('GET', f'{self.base_url}/multi_headers', headers={**headers, 'Extra': 'extra', 'Foo': 'new'})
            returned_headers = r.json()['headers']
            assert returned_headers[-4:] == [['Foo', 'new'], ['Multi', '1, 2'], ['Baz', 'quux'], ['Extra', 'extra']]

    def test_merge_headers_with_pool_manager_headers(self) -> None:
        if False:
            return 10
        headers = HTTPHeaderDict()
        headers.add('Cookie', 'choc-chip')
        headers.add('Cookie', 'oatmeal-raisin')
        orig = headers.copy()
        added_headers = {'Cookie': 'tim-tam'}
        with PoolManager(headers=headers) as http:
            r = http.request('GET', f'{self.base_url}/multi_headers', headers=typing.cast(HTTPHeaderDict, http.headers) | added_headers)
            returned_headers = r.json()['headers']
            assert returned_headers[-3:] == [['Cookie', 'choc-chip'], ['Cookie', 'oatmeal-raisin'], ['Cookie', 'tim-tam']]
            assert http.headers == orig

    def test_headers_http_multi_header_multipart(self) -> None:
        if False:
            print('Hello World!')
        headers = HTTPHeaderDict()
        headers.add('Multi', '1')
        headers.add('Multi', '2')
        old_headers = headers.copy()
        with PoolManager(headers=headers) as http:
            r = http.request('POST', f'{self.base_url}/multi_headers', fields={'k': 'v'}, multipart_boundary='b', encode_multipart=True)
            returned_headers = r.json()['headers']
            assert returned_headers[4:] == [['Multi', '1'], ['Multi', '2'], ['Content-Type', 'multipart/form-data; boundary=b']]
            assert headers == old_headers
            headers['Content-Type'] = 'multipart/form-data; boundary=b; field=value'
            r = http.request('POST', f'{self.base_url}/multi_headers', fields={'k': 'v'}, multipart_boundary='b', encode_multipart=True)
            returned_headers = r.json()['headers']
            assert returned_headers[4:] == [['Multi', '1'], ['Multi', '2'], ['Content-Type', 'multipart/form-data; boundary=b; field=value']]

    def test_body(self) -> None:
        if False:
            return 10
        with PoolManager() as http:
            r = http.request('POST', f'{self.base_url}/echo', body=b'test')
            assert r.data == b'test'

    def test_http_with_ssl_keywords(self) -> None:
        if False:
            return 10
        with PoolManager(ca_certs='REQUIRED') as http:
            r = http.request('GET', f'http://{self.host}:{self.port}/')
            assert r.status == 200

    def test_http_with_server_hostname(self) -> None:
        if False:
            i = 10
            return i + 15
        with PoolManager(server_hostname='example.com') as http:
            r = http.request('GET', f'http://{self.host}:{self.port}/')
            assert r.status == 200

    def test_http_with_ca_cert_dir(self) -> None:
        if False:
            i = 10
            return i + 15
        with PoolManager(ca_certs='REQUIRED', ca_cert_dir='/nosuchdir') as http:
            r = http.request('GET', f'http://{self.host}:{self.port}/')
            assert r.status == 200

    @pytest.mark.parametrize(['target', 'expected_target'], [('/echo_uri?q=1#fragment', b'/echo_uri?q=1'), ('/echo_uri?#', b'/echo_uri?'), ('/echo_uri#?', b'/echo_uri'), ('/echo_uri#?#', b'/echo_uri'), ('/echo_uri??#', b'/echo_uri??'), ('/echo_uri?%3f#', b'/echo_uri?%3F'), ('/echo_uri?%3F#', b'/echo_uri?%3F'), ('/echo_uri?[]', b'/echo_uri?%5B%5D')])
    def test_encode_http_target(self, target: str, expected_target: bytes) -> None:
        if False:
            while True:
                i = 10
        with PoolManager() as http:
            url = f'http://{self.host}:{self.port}{target}'
            r = http.request('GET', url)
            assert r.data == expected_target

    def test_top_level_request(self) -> None:
        if False:
            print('Hello World!')
        r = request('GET', f'{self.base_url}/')
        assert r.status == 200
        assert r.data == b'Dummy server!'

    def test_top_level_request_without_keyword_args(self) -> None:
        if False:
            while True:
                i = 10
        body = ''
        with pytest.raises(TypeError):
            request('GET', f'{self.base_url}/', body)

    def test_top_level_request_with_body(self) -> None:
        if False:
            print('Hello World!')
        r = request('POST', f'{self.base_url}/echo', body=b'test')
        assert r.status == 200
        assert r.data == b'test'

    def test_top_level_request_with_preload_content(self) -> None:
        if False:
            return 10
        r = request('GET', f'{self.base_url}/echo', preload_content=False)
        assert r.status == 200
        assert r.connection is not None
        r.data
        assert r.connection is None

    def test_top_level_request_with_decode_content(self) -> None:
        if False:
            i = 10
            return i + 15
        r = request('GET', f'{self.base_url}/encodingrequest', headers={'accept-encoding': 'gzip'}, decode_content=False)
        assert r.status == 200
        assert gzip.decompress(r.data) == b'hello, world!'
        r = request('GET', f'{self.base_url}/encodingrequest', headers={'accept-encoding': 'gzip'}, decode_content=True)
        assert r.status == 200
        assert r.data == b'hello, world!'

    def test_top_level_request_with_redirect(self) -> None:
        if False:
            while True:
                i = 10
        r = request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url}/'}, redirect=False)
        assert r.status == 303
        r = request('GET', f'{self.base_url}/redirect', fields={'target': f'{self.base_url}/'}, redirect=True)
        assert r.status == 200
        assert r.data == b'Dummy server!'

    def test_top_level_request_with_retries(self) -> None:
        if False:
            i = 10
            return i + 15
        r = request('GET', f'{self.base_url}/redirect', retries=False)
        assert r.status == 303
        r = request('GET', f'{self.base_url}/redirect', retries=3)
        assert r.status == 200

    def test_top_level_request_with_timeout(self) -> None:
        if False:
            i = 10
            return i + 15
        with mock.patch('urllib3.poolmanager.RequestMethods.request') as mockRequest:
            mockRequest.return_value = HTTPResponse(status=200)
            r = request('GET', f'{self.base_url}/redirect', timeout=2.5)
            assert r.status == 200
            mockRequest.assert_called_with('GET', f'{self.base_url}/redirect', body=None, fields=None, headers=None, preload_content=True, decode_content=True, redirect=True, retries=None, timeout=2.5, json=None)

    @pytest.mark.parametrize('headers', [None, {'content-Type': 'application/json'}, {'content-Type': 'text/plain'}, {'attribute': 'value', 'CONTENT-TYPE': 'application/json'}, HTTPHeaderDict(cookie='foo, bar')])
    def test_request_with_json(self, headers: HTTPHeaderDict) -> None:
        if False:
            return 10
        body = {'attribute': 'value'}
        r = request(method='POST', url=f'{self.base_url}/echo_json', headers=headers, json=body)
        assert r.status == 200
        assert r.json() == body
        if headers is not None and 'application/json' not in headers.values():
            assert 'text/plain' in r.headers['Content-Type'].replace(' ', '').split(',')
        else:
            assert 'application/json' in r.headers['Content-Type'].replace(' ', '').split(',')

    def test_top_level_request_with_json_with_httpheaderdict(self) -> None:
        if False:
            i = 10
            return i + 15
        body = {'attribute': 'value'}
        header = HTTPHeaderDict(cookie='foo, bar')
        with PoolManager(headers=header) as http:
            r = http.request(method='POST', url=f'{self.base_url}/echo_json', json=body)
            assert r.status == 200
            assert r.json() == body
            assert 'application/json' in r.headers['Content-Type'].replace(' ', '').split(',')

    def test_top_level_request_with_body_and_json(self) -> None:
        if False:
            print('Hello World!')
        match = "request got values for both 'body' and 'json' parameters which are mutually exclusive"
        with pytest.raises(TypeError, match=match):
            body = {'attribute': 'value'}
            request(method='POST', url=f'{self.base_url}/echo', body='', json=body)

    def test_top_level_request_with_invalid_body(self) -> None:
        if False:
            i = 10
            return i + 15

        class BadBody:

            def __repr__(self) -> str:
                if False:
                    return 10
                return '<BadBody>'
        with pytest.raises(TypeError) as e:
            request(method='POST', url=f'{self.base_url}/echo', body=BadBody())
        assert str(e.value) == "'body' must be a bytes-like object, file-like object, or iterable. Instead was <BadBody>"

@pytest.mark.skipif(not HAS_IPV6, reason='IPv6 is not supported on this system')
class TestIPv6PoolManager(IPv6HTTPDummyServerTestCase):

    @classmethod
    def setup_class(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setup_class()
        cls.base_url = f'http://[{cls.host}]:{cls.port}'

    def test_ipv6(self) -> None:
        if False:
            while True:
                i = 10
        with PoolManager() as http:
            http.request('GET', self.base_url)