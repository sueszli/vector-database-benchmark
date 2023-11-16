import errno
from http import client, HTTPStatus
import io
import itertools
import os
import array
import re
import socket
import threading
import warnings
import unittest
from unittest import mock
TestCase = unittest.TestCase
from test import support
from test.support import socket_helper
here = os.path.dirname(__file__)
CERT_localhost = os.path.join(here, 'keycert.pem')
CERT_fakehostname = os.path.join(here, 'keycert2.pem')
CERT_selfsigned_pythontestdotnet = os.path.join(here, 'selfsigned_pythontestdotnet.pem')
chunked_start = 'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\na\r\nhello worl\r\n3\r\nd! \r\n8\r\nand now \r\n22\r\nfor something completely different\r\n'
chunked_expected = b'hello world! and now for something completely different'
chunk_extension = ';foo=bar'
last_chunk = '0\r\n'
last_chunk_extended = '0' + chunk_extension + '\r\n'
trailers = 'X-Dummy: foo\r\nX-Dumm2: bar\r\n'
chunked_end = '\r\n'
HOST = socket_helper.HOST

class FakeSocket:

    def __init__(self, text, fileclass=io.BytesIO, host=None, port=None):
        if False:
            return 10
        if isinstance(text, str):
            text = text.encode('ascii')
        self.text = text
        self.fileclass = fileclass
        self.data = b''
        self.sendall_calls = 0
        self.file_closed = False
        self.host = host
        self.port = port

    def sendall(self, data):
        if False:
            print('Hello World!')
        self.sendall_calls += 1
        self.data += data

    def makefile(self, mode, bufsize=None):
        if False:
            for i in range(10):
                print('nop')
        if mode != 'r' and mode != 'rb':
            raise client.UnimplementedFileMode()
        self.file = self.fileclass(self.text)
        self.file.close = self.file_close
        return self.file

    def file_close(self):
        if False:
            print('Hello World!')
        self.file_closed = True

    def close(self):
        if False:
            return 10
        pass

    def setsockopt(self, level, optname, value):
        if False:
            print('Hello World!')
        pass

class EPipeSocket(FakeSocket):

    def __init__(self, text, pipe_trigger):
        if False:
            print('Hello World!')
        FakeSocket.__init__(self, text)
        self.pipe_trigger = pipe_trigger

    def sendall(self, data):
        if False:
            i = 10
            return i + 15
        if self.pipe_trigger in data:
            raise OSError(errno.EPIPE, 'gotcha')
        self.data += data

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class NoEOFBytesIO(io.BytesIO):
    """Like BytesIO, but raises AssertionError on EOF.

    This is used below to test that http.client doesn't try to read
    more from the underlying file than it should.
    """

    def read(self, n=-1):
        if False:
            print('Hello World!')
        data = io.BytesIO.read(self, n)
        if data == b'':
            raise AssertionError('caller tried to read past EOF')
        return data

    def readline(self, length=None):
        if False:
            for i in range(10):
                print('nop')
        data = io.BytesIO.readline(self, length)
        if data == b'':
            raise AssertionError('caller tried to read past EOF')
        return data

class FakeSocketHTTPConnection(client.HTTPConnection):
    """HTTPConnection subclass using FakeSocket; counts connect() calls"""

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        self.connections = 0
        super().__init__('example.com')
        self.fake_socket_args = args
        self._create_connection = self.create_connection

    def connect(self):
        if False:
            return 10
        'Count the number of times connect() is invoked'
        self.connections += 1
        return super().connect()

    def create_connection(self, *pos, **kw):
        if False:
            i = 10
            return i + 15
        return FakeSocket(*self.fake_socket_args)

class HeaderTests(TestCase):

    def test_auto_headers(self):
        if False:
            for i in range(10):
                print('nop')

        class HeaderCountingBuffer(list):

            def __init__(self):
                if False:
                    return 10
                self.count = {}

            def append(self, item):
                if False:
                    while True:
                        i = 10
                kv = item.split(b':')
                if len(kv) > 1:
                    lcKey = kv[0].decode('ascii').lower()
                    self.count.setdefault(lcKey, 0)
                    self.count[lcKey] += 1
                list.append(self, item)
        for explicit_header in (True, False):
            for header in ('Content-length', 'Host', 'Accept-encoding'):
                conn = client.HTTPConnection('example.com')
                conn.sock = FakeSocket('blahblahblah')
                conn._buffer = HeaderCountingBuffer()
                body = 'spamspamspam'
                headers = {}
                if explicit_header:
                    headers[header] = str(len(body))
                conn.request('POST', '/', body, headers)
                self.assertEqual(conn._buffer.count[header.lower()], 1)

    def test_content_length_0(self):
        if False:
            while True:
                i = 10

        class ContentLengthChecker(list):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                list.__init__(self)
                self.content_length = None

            def append(self, item):
                if False:
                    return 10
                kv = item.split(b':', 1)
                if len(kv) > 1 and kv[0].lower() == b'content-length':
                    self.content_length = kv[1].strip()
                list.append(self, item)
        bodies = (None, '')
        methods_with_body = ('PUT', 'POST', 'PATCH')
        for (method, body) in itertools.product(methods_with_body, bodies):
            conn = client.HTTPConnection('example.com')
            conn.sock = FakeSocket(None)
            conn._buffer = ContentLengthChecker()
            conn.request(method, '/', body)
            self.assertEqual(conn._buffer.content_length, b'0', 'Header Content-Length incorrect on {}'.format(method))
        methods_without_body = ('GET', 'CONNECT', 'DELETE', 'HEAD', 'OPTIONS', 'TRACE')
        for method in methods_without_body:
            conn = client.HTTPConnection('example.com')
            conn.sock = FakeSocket(None)
            conn._buffer = ContentLengthChecker()
            conn.request(method, '/', None)
            self.assertEqual(conn._buffer.content_length, None, 'Header Content-Length set for empty body on {}'.format(method))
        for method in methods_without_body:
            conn = client.HTTPConnection('example.com')
            conn.sock = FakeSocket(None)
            conn._buffer = ContentLengthChecker()
            conn.request(method, '/', '')
            self.assertEqual(conn._buffer.content_length, b'0', 'Header Content-Length incorrect on {}'.format(method))
        for method in itertools.chain(methods_without_body, methods_with_body):
            conn = client.HTTPConnection('example.com')
            conn.sock = FakeSocket(None)
            conn._buffer = ContentLengthChecker()
            conn.request(method, '/', ' ')
            self.assertEqual(conn._buffer.content_length, b'1', 'Header Content-Length incorrect on {}'.format(method))

    def test_putheader(self):
        if False:
            print('Hello World!')
        conn = client.HTTPConnection('example.com')
        conn.sock = FakeSocket(None)
        conn.putrequest('GET', '/')
        conn.putheader('Content-length', 42)
        self.assertIn(b'Content-length: 42', conn._buffer)
        conn.putheader('Foo', ' bar ')
        self.assertIn(b'Foo:  bar ', conn._buffer)
        conn.putheader('Bar', '\tbaz\t')
        self.assertIn(b'Bar: \tbaz\t', conn._buffer)
        conn.putheader('Authorization', 'Bearer mytoken')
        self.assertIn(b'Authorization: Bearer mytoken', conn._buffer)
        conn.putheader('IterHeader', 'IterA', 'IterB')
        self.assertIn(b'IterHeader: IterA\r\n\tIterB', conn._buffer)
        conn.putheader('LatinHeader', b'\xff')
        self.assertIn(b'LatinHeader: \xff', conn._buffer)
        conn.putheader('Utf8Header', b'\xc3\x80')
        self.assertIn(b'Utf8Header: \xc3\x80', conn._buffer)
        conn.putheader('C1-Control', b'next\x85line')
        self.assertIn(b'C1-Control: next\x85line', conn._buffer)
        conn.putheader('Embedded-Fold-Space', 'is\r\n allowed')
        self.assertIn(b'Embedded-Fold-Space: is\r\n allowed', conn._buffer)
        conn.putheader('Embedded-Fold-Tab', 'is\r\n\tallowed')
        self.assertIn(b'Embedded-Fold-Tab: is\r\n\tallowed', conn._buffer)
        conn.putheader('Key Space', 'value')
        self.assertIn(b'Key Space: value', conn._buffer)
        conn.putheader('KeySpace ', 'value')
        self.assertIn(b'KeySpace : value', conn._buffer)
        conn.putheader(b'Nonbreak\xa0Space', 'value')
        self.assertIn(b'Nonbreak\xa0Space: value', conn._buffer)
        conn.putheader(b'\xa0NonbreakSpace', 'value')
        self.assertIn(b'\xa0NonbreakSpace: value', conn._buffer)

    def test_ipv6host_header(self):
        if False:
            for i in range(10):
                print('nop')
        expected = b'GET /foo HTTP/1.1\r\nHost: [2001::]:81\r\nAccept-Encoding: identity\r\n\r\n'
        conn = client.HTTPConnection('[2001::]:81')
        sock = FakeSocket('')
        conn.sock = sock
        conn.request('GET', '/foo')
        self.assertTrue(sock.data.startswith(expected))
        expected = b'GET /foo HTTP/1.1\r\nHost: [2001:102A::]\r\nAccept-Encoding: identity\r\n\r\n'
        conn = client.HTTPConnection('[2001:102A::]')
        sock = FakeSocket('')
        conn.sock = sock
        conn.request('GET', '/foo')
        self.assertTrue(sock.data.startswith(expected))

    def test_malformed_headers_coped_with(self):
        if False:
            while True:
                i = 10
        body = 'HTTP/1.1 200 OK\r\nFirst: val\r\n: nval\r\nSecond: val\r\n\r\n'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock)
        resp.begin()
        self.assertEqual(resp.getheader('First'), 'val')
        self.assertEqual(resp.getheader('Second'), 'val')

    def test_parse_all_octets(self):
        if False:
            while True:
                i = 10
        body = b"HTTP/1.1 200 OK\r\n!#$%&'*+-.^_`|~: value\r\nVCHAR: " + bytes(range(33, 126 + 1)) + b'\r\nobs-text: ' + bytes(range(128, 255 + 1)) + b'\r\nobs-fold: text\r\n folded with space\r\n\tfolded with tab\r\nContent-Length: 0\r\n\r\n'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock)
        resp.begin()
        self.assertEqual(resp.getheader('Content-Length'), '0')
        self.assertEqual(resp.msg['Content-Length'], '0')
        self.assertEqual(resp.getheader("!#$%&'*+-.^_`|~"), 'value')
        self.assertEqual(resp.msg["!#$%&'*+-.^_`|~"], 'value')
        vchar = ''.join(map(chr, range(33, 126 + 1)))
        self.assertEqual(resp.getheader('VCHAR'), vchar)
        self.assertEqual(resp.msg['VCHAR'], vchar)
        self.assertIsNotNone(resp.getheader('obs-text'))
        self.assertIn('obs-text', resp.msg)
        for folded in (resp.getheader('obs-fold'), resp.msg['obs-fold']):
            self.assertTrue(folded.startswith('text'))
            self.assertIn(' folded with space', folded)
            self.assertTrue(folded.endswith('folded with tab'))

    def test_invalid_headers(self):
        if False:
            return 10
        conn = client.HTTPConnection('example.com')
        conn.sock = FakeSocket('')
        conn.putrequest('GET', '/')
        cases = ((b'Invalid\r\nName', b'ValidValue'), (b'Invalid\rName', b'ValidValue'), (b'Invalid\nName', b'ValidValue'), (b'\r\nInvalidName', b'ValidValue'), (b'\rInvalidName', b'ValidValue'), (b'\nInvalidName', b'ValidValue'), (b' InvalidName', b'ValidValue'), (b'\tInvalidName', b'ValidValue'), (b'Invalid:Name', b'ValidValue'), (b':InvalidName', b'ValidValue'), (b'ValidName', b'Invalid\r\nValue'), (b'ValidName', b'Invalid\rValue'), (b'ValidName', b'Invalid\nValue'), (b'ValidName', b'InvalidValue\r\n'), (b'ValidName', b'InvalidValue\r'), (b'ValidName', b'InvalidValue\n'))
        for (name, value) in cases:
            with self.subTest((name, value)):
                with self.assertRaisesRegex(ValueError, 'Invalid header'):
                    conn.putheader(name, value)

    def test_headers_debuglevel(self):
        if False:
            return 10
        body = b'HTTP/1.1 200 OK\r\nFirst: val\r\nSecond: val1\r\nSecond: val2\r\n'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock, debuglevel=1)
        with support.captured_stdout() as output:
            resp.begin()
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[0], "reply: 'HTTP/1.1 200 OK\\r\\n'")
        self.assertEqual(lines[1], 'header: First: val')
        self.assertEqual(lines[2], 'header: Second: val1')
        self.assertEqual(lines[3], 'header: Second: val2')

class HttpMethodTests(TestCase):

    def test_invalid_method_names(self):
        if False:
            i = 10
            return i + 15
        methods = ('GET\r', 'POST\n', 'PUT\n\r', 'POST\nValue', 'POST\nHOST:abc', 'GET\nrHost:abc\n', 'POST\rRemainder:\r', 'GET\rHOST:\n', '\nPUT')
        for method in methods:
            with self.assertRaisesRegex(ValueError, "method can't contain control characters"):
                conn = client.HTTPConnection('example.com')
                conn.sock = FakeSocket(None)
                conn.request(method=method, url='/')

class TransferEncodingTest(TestCase):
    expected_body = b"It's just a flesh wound"

    def test_endheaders_chunked(self):
        if False:
            print('Hello World!')
        conn = client.HTTPConnection('example.com')
        conn.sock = FakeSocket(b'')
        conn.putrequest('POST', '/')
        conn.endheaders(self._make_body(), encode_chunked=True)
        (_, _, body) = self._parse_request(conn.sock.data)
        body = self._parse_chunked(body)
        self.assertEqual(body, self.expected_body)

    def test_explicit_headers(self):
        if False:
            return 10
        conn = client.HTTPConnection('example.com')
        conn.sock = FakeSocket(b'')
        conn.request('POST', '/', self._make_body(), {'Transfer-Encoding': 'chunked'})
        (_, headers, body) = self._parse_request(conn.sock.data)
        self.assertNotIn('content-length', [k.lower() for k in headers.keys()])
        self.assertEqual(headers['Transfer-Encoding'], 'chunked')
        self.assertEqual(body, self.expected_body)
        conn = client.HTTPConnection('example.com')
        conn.sock = FakeSocket(b'')
        conn.request('POST', '/', self.expected_body.decode('latin-1'), {'Transfer-Encoding': 'chunked'})
        (_, headers, body) = self._parse_request(conn.sock.data)
        self.assertNotIn('content-length', [k.lower() for k in headers.keys()])
        self.assertEqual(headers['Transfer-Encoding'], 'chunked')
        self.assertEqual(body, self.expected_body)
        conn = client.HTTPConnection('example.com')
        conn.sock = FakeSocket(b'')
        conn.request('POST', '/', headers={'Transfer-Encoding': 'gzip, chunked'}, encode_chunked=True, body=self._make_body())
        (_, headers, body) = self._parse_request(conn.sock.data)
        self.assertNotIn('content-length', [k.lower() for k in headers])
        self.assertEqual(headers['Transfer-Encoding'], 'gzip, chunked')
        self.assertEqual(self._parse_chunked(body), self.expected_body)

    def test_request(self):
        if False:
            i = 10
            return i + 15
        for empty_lines in (False, True):
            conn = client.HTTPConnection('example.com')
            conn.sock = FakeSocket(b'')
            conn.request('POST', '/', self._make_body(empty_lines=empty_lines))
            (_, headers, body) = self._parse_request(conn.sock.data)
            body = self._parse_chunked(body)
            self.assertEqual(body, self.expected_body)
            self.assertEqual(headers['Transfer-Encoding'], 'chunked')
            self.assertNotIn('content-length', [k.lower() for k in headers])

    def test_empty_body(self):
        if False:
            i = 10
            return i + 15
        conn = client.HTTPConnection('example.com')
        conn.sock = FakeSocket(b'')
        conn.request('POST', '/', ())
        (_, headers, body) = self._parse_request(conn.sock.data)
        self.assertEqual(headers['Transfer-Encoding'], 'chunked')
        self.assertNotIn('content-length', [k.lower() for k in headers])
        self.assertEqual(body, b'0\r\n\r\n')

    def _make_body(self, empty_lines=False):
        if False:
            for i in range(10):
                print('nop')
        lines = self.expected_body.split(b' ')
        for (idx, line) in enumerate(lines):
            if empty_lines and idx % 2:
                yield b''
            if idx < len(lines) - 1:
                yield (line + b' ')
            else:
                yield line

    def _parse_request(self, data):
        if False:
            return 10
        lines = data.split(b'\r\n')
        request = lines[0]
        headers = {}
        n = 1
        while n < len(lines) and len(lines[n]) > 0:
            (key, val) = lines[n].split(b':')
            key = key.decode('latin-1').strip()
            headers[key] = val.decode('latin-1').strip()
            n += 1
        return (request, headers, b'\r\n'.join(lines[n + 1:]))

    def _parse_chunked(self, data):
        if False:
            for i in range(10):
                print('nop')
        body = []
        trailers = {}
        n = 0
        lines = data.split(b'\r\n')
        while True:
            (size, chunk) = lines[n:n + 2]
            size = int(size, 16)
            if size == 0:
                n += 1
                break
            self.assertEqual(size, len(chunk))
            body.append(chunk)
            n += 2
            if n > len(lines):
                break
        return b''.join(body)

class BasicTest(TestCase):

    def test_dir_with_added_behavior_on_status(self):
        if False:
            while True:
                i = 10
        self.assertTrue({'description', 'name', 'phrase', 'value'} <= set(dir(HTTPStatus(404))))

    def test_status_lines(self):
        if False:
            print('Hello World!')
        body = 'HTTP/1.1 200 Ok\r\n\r\nText'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock)
        resp.begin()
        self.assertEqual(resp.read(0), b'')
        self.assertFalse(resp.isclosed())
        self.assertFalse(resp.closed)
        self.assertEqual(resp.read(), b'Text')
        self.assertTrue(resp.isclosed())
        self.assertFalse(resp.closed)
        resp.close()
        self.assertTrue(resp.closed)
        body = 'HTTP/1.1 400.100 Not Ok\r\n\r\nText'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock)
        self.assertRaises(client.BadStatusLine, resp.begin)

    def test_bad_status_repr(self):
        if False:
            return 10
        exc = client.BadStatusLine('')
        self.assertEqual(repr(exc), 'BadStatusLine("\'\'")')

    def test_partial_reads(self):
        if False:
            for i in range(10):
                print('nop')
        body = 'HTTP/1.1 200 Ok\r\nContent-Length: 4\r\n\r\nText'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock)
        resp.begin()
        self.assertEqual(resp.read(2), b'Te')
        self.assertFalse(resp.isclosed())
        self.assertEqual(resp.read(2), b'xt')
        self.assertTrue(resp.isclosed())
        self.assertFalse(resp.closed)
        resp.close()
        self.assertTrue(resp.closed)

    def test_mixed_reads(self):
        if False:
            print('Hello World!')
        body = 'HTTP/1.1 200 Ok\r\nContent-Length: 13\r\n\r\nText\r\nAnother'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock)
        resp.begin()
        self.assertEqual(resp.readline(), b'Text\r\n')
        self.assertFalse(resp.isclosed())
        self.assertEqual(resp.read(), b'Another')
        self.assertTrue(resp.isclosed())
        self.assertFalse(resp.closed)
        resp.close()
        self.assertTrue(resp.closed)

    def test_partial_readintos(self):
        if False:
            for i in range(10):
                print('nop')
        body = 'HTTP/1.1 200 Ok\r\nContent-Length: 4\r\n\r\nText'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock)
        resp.begin()
        b = bytearray(2)
        n = resp.readinto(b)
        self.assertEqual(n, 2)
        self.assertEqual(bytes(b), b'Te')
        self.assertFalse(resp.isclosed())
        n = resp.readinto(b)
        self.assertEqual(n, 2)
        self.assertEqual(bytes(b), b'xt')
        self.assertTrue(resp.isclosed())
        self.assertFalse(resp.closed)
        resp.close()
        self.assertTrue(resp.closed)

    def test_partial_reads_no_content_length(self):
        if False:
            print('Hello World!')
        body = 'HTTP/1.1 200 Ok\r\n\r\nText'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock)
        resp.begin()
        self.assertEqual(resp.read(2), b'Te')
        self.assertFalse(resp.isclosed())
        self.assertEqual(resp.read(2), b'xt')
        self.assertEqual(resp.read(1), b'')
        self.assertTrue(resp.isclosed())
        self.assertFalse(resp.closed)
        resp.close()
        self.assertTrue(resp.closed)

    def test_partial_readintos_no_content_length(self):
        if False:
            return 10
        body = 'HTTP/1.1 200 Ok\r\n\r\nText'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock)
        resp.begin()
        b = bytearray(2)
        n = resp.readinto(b)
        self.assertEqual(n, 2)
        self.assertEqual(bytes(b), b'Te')
        self.assertFalse(resp.isclosed())
        n = resp.readinto(b)
        self.assertEqual(n, 2)
        self.assertEqual(bytes(b), b'xt')
        n = resp.readinto(b)
        self.assertEqual(n, 0)
        self.assertTrue(resp.isclosed())

    def test_partial_reads_incomplete_body(self):
        if False:
            i = 10
            return i + 15
        body = 'HTTP/1.1 200 Ok\r\nContent-Length: 10\r\n\r\nText'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock)
        resp.begin()
        self.assertEqual(resp.read(2), b'Te')
        self.assertFalse(resp.isclosed())
        self.assertEqual(resp.read(2), b'xt')
        self.assertEqual(resp.read(1), b'')
        self.assertTrue(resp.isclosed())

    def test_partial_readintos_incomplete_body(self):
        if False:
            print('Hello World!')
        body = 'HTTP/1.1 200 Ok\r\nContent-Length: 10\r\n\r\nText'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock)
        resp.begin()
        b = bytearray(2)
        n = resp.readinto(b)
        self.assertEqual(n, 2)
        self.assertEqual(bytes(b), b'Te')
        self.assertFalse(resp.isclosed())
        n = resp.readinto(b)
        self.assertEqual(n, 2)
        self.assertEqual(bytes(b), b'xt')
        n = resp.readinto(b)
        self.assertEqual(n, 0)
        self.assertTrue(resp.isclosed())
        self.assertFalse(resp.closed)
        resp.close()
        self.assertTrue(resp.closed)

    def test_host_port(self):
        if False:
            print('Hello World!')
        for hp in ('www.python.org:abc', 'user:password@www.python.org'):
            self.assertRaises(client.InvalidURL, client.HTTPConnection, hp)
        for (hp, h, p) in (('[fe80::207:e9ff:fe9b]:8000', 'fe80::207:e9ff:fe9b', 8000), ('www.python.org:80', 'www.python.org', 80), ('www.python.org:', 'www.python.org', 80), ('www.python.org', 'www.python.org', 80), ('[fe80::207:e9ff:fe9b]', 'fe80::207:e9ff:fe9b', 80), ('[fe80::207:e9ff:fe9b]:', 'fe80::207:e9ff:fe9b', 80)):
            c = client.HTTPConnection(hp)
            self.assertEqual(h, c.host)
            self.assertEqual(p, c.port)

    def test_response_headers(self):
        if False:
            return 10
        text = 'HTTP/1.1 200 OK\r\nSet-Cookie: Customer="WILE_E_COYOTE"; Version="1"; Path="/acme"\r\nSet-Cookie: Part_Number="Rocket_Launcher_0001"; Version="1"; Path="/acme"\r\n\r\nNo body\r\n'
        hdr = 'Customer="WILE_E_COYOTE"; Version="1"; Path="/acme", Part_Number="Rocket_Launcher_0001"; Version="1"; Path="/acme"'
        s = FakeSocket(text)
        r = client.HTTPResponse(s)
        r.begin()
        cookies = r.getheader('Set-Cookie')
        self.assertEqual(cookies, hdr)

    def test_read_head(self):
        if False:
            return 10
        sock = FakeSocket('HTTP/1.1 200 OK\r\nContent-Length: 14432\r\n\r\n', NoEOFBytesIO)
        resp = client.HTTPResponse(sock, method='HEAD')
        resp.begin()
        if resp.read():
            self.fail('Did not expect response from HEAD request')

    def test_readinto_head(self):
        if False:
            for i in range(10):
                print('nop')
        sock = FakeSocket('HTTP/1.1 200 OK\r\nContent-Length: 14432\r\n\r\n', NoEOFBytesIO)
        resp = client.HTTPResponse(sock, method='HEAD')
        resp.begin()
        b = bytearray(5)
        if resp.readinto(b) != 0:
            self.fail('Did not expect response from HEAD request')
        self.assertEqual(bytes(b), b'\x00' * 5)

    def test_too_many_headers(self):
        if False:
            for i in range(10):
                print('nop')
        headers = '\r\n'.join(('Header%d: foo' % i for i in range(client._MAXHEADERS + 1))) + '\r\n'
        text = 'HTTP/1.1 200 OK\r\n' + headers
        s = FakeSocket(text)
        r = client.HTTPResponse(s)
        self.assertRaisesRegex(client.HTTPException, 'got more than \\d+ headers', r.begin)

    def test_send_file(self):
        if False:
            print('Hello World!')
        expected = b'GET /foo HTTP/1.1\r\nHost: example.com\r\nAccept-Encoding: identity\r\nTransfer-Encoding: chunked\r\n\r\n'
        with open(__file__, 'rb') as body:
            conn = client.HTTPConnection('example.com')
            sock = FakeSocket(body)
            conn.sock = sock
            conn.request('GET', '/foo', body)
            self.assertTrue(sock.data.startswith(expected), '%r != %r' % (sock.data[:len(expected)], expected))

    def test_send(self):
        if False:
            return 10
        expected = b'this is a test this is only a test'
        conn = client.HTTPConnection('example.com')
        sock = FakeSocket(None)
        conn.sock = sock
        conn.send(expected)
        self.assertEqual(expected, sock.data)
        sock.data = b''
        conn.send(array.array('b', expected))
        self.assertEqual(expected, sock.data)
        sock.data = b''
        conn.send(io.BytesIO(expected))
        self.assertEqual(expected, sock.data)

    def test_send_updating_file(self):
        if False:
            i = 10
            return i + 15

        def data():
            if False:
                while True:
                    i = 10
            yield 'data'
            yield None
            yield 'data_two'

        class UpdatingFile(io.TextIOBase):
            mode = 'r'
            d = data()

            def read(self, blocksize=-1):
                if False:
                    print('Hello World!')
                return next(self.d)
        expected = b'data'
        conn = client.HTTPConnection('example.com')
        sock = FakeSocket('')
        conn.sock = sock
        conn.send(UpdatingFile())
        self.assertEqual(sock.data, expected)

    def test_send_iter(self):
        if False:
            for i in range(10):
                print('nop')
        expected = b'GET /foo HTTP/1.1\r\nHost: example.com\r\nAccept-Encoding: identity\r\nContent-Length: 11\r\n\r\nonetwothree'

        def body():
            if False:
                while True:
                    i = 10
            yield b'one'
            yield b'two'
            yield b'three'
        conn = client.HTTPConnection('example.com')
        sock = FakeSocket('')
        conn.sock = sock
        conn.request('GET', '/foo', body(), {'Content-Length': '11'})
        self.assertEqual(sock.data, expected)

    def test_blocksize_request(self):
        if False:
            i = 10
            return i + 15
        'Check that request() respects the configured block size.'
        blocksize = 8
        conn = client.HTTPConnection('example.com', blocksize=blocksize)
        sock = FakeSocket(None)
        conn.sock = sock
        expected = b'a' * blocksize + b'b'
        conn.request('PUT', '/', io.BytesIO(expected), {'Content-Length': '9'})
        self.assertEqual(sock.sendall_calls, 3)
        body = sock.data.split(b'\r\n\r\n', 1)[1]
        self.assertEqual(body, expected)

    def test_blocksize_send(self):
        if False:
            return 10
        'Check that send() respects the configured block size.'
        blocksize = 8
        conn = client.HTTPConnection('example.com', blocksize=blocksize)
        sock = FakeSocket(None)
        conn.sock = sock
        expected = b'a' * blocksize + b'b'
        conn.send(io.BytesIO(expected))
        self.assertEqual(sock.sendall_calls, 2)
        self.assertEqual(sock.data, expected)

    def test_send_type_error(self):
        if False:
            print('Hello World!')
        conn = client.HTTPConnection('example.com')
        conn.sock = FakeSocket('')
        with self.assertRaises(TypeError):
            conn.request('POST', 'test', conn)

    def test_chunked(self):
        if False:
            i = 10
            return i + 15
        expected = chunked_expected
        sock = FakeSocket(chunked_start + last_chunk + chunked_end)
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        self.assertEqual(resp.read(), expected)
        resp.close()
        for n in range(1, 12):
            sock = FakeSocket(chunked_start + last_chunk + chunked_end)
            resp = client.HTTPResponse(sock, method='GET')
            resp.begin()
            self.assertEqual(resp.read(n) + resp.read(n) + resp.read(), expected)
            resp.close()
        for x in ('', 'foo\r\n'):
            sock = FakeSocket(chunked_start + x)
            resp = client.HTTPResponse(sock, method='GET')
            resp.begin()
            try:
                resp.read()
            except client.IncompleteRead as i:
                self.assertEqual(i.partial, expected)
                expected_message = 'IncompleteRead(%d bytes read)' % len(expected)
                self.assertEqual(repr(i), expected_message)
                self.assertEqual(str(i), expected_message)
            else:
                self.fail('IncompleteRead expected')
            finally:
                resp.close()

    def test_readinto_chunked(self):
        if False:
            print('Hello World!')
        expected = chunked_expected
        nexpected = len(expected)
        b = bytearray(128)
        sock = FakeSocket(chunked_start + last_chunk + chunked_end)
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        n = resp.readinto(b)
        self.assertEqual(b[:nexpected], expected)
        self.assertEqual(n, nexpected)
        resp.close()
        for n in range(1, 12):
            sock = FakeSocket(chunked_start + last_chunk + chunked_end)
            resp = client.HTTPResponse(sock, method='GET')
            resp.begin()
            m = memoryview(b)
            i = resp.readinto(m[0:n])
            i += resp.readinto(m[i:n + i])
            i += resp.readinto(m[i:])
            self.assertEqual(b[:nexpected], expected)
            self.assertEqual(i, nexpected)
            resp.close()
        for x in ('', 'foo\r\n'):
            sock = FakeSocket(chunked_start + x)
            resp = client.HTTPResponse(sock, method='GET')
            resp.begin()
            try:
                n = resp.readinto(b)
            except client.IncompleteRead as i:
                self.assertEqual(i.partial, expected)
                expected_message = 'IncompleteRead(%d bytes read)' % len(expected)
                self.assertEqual(repr(i), expected_message)
                self.assertEqual(str(i), expected_message)
            else:
                self.fail('IncompleteRead expected')
            finally:
                resp.close()

    def test_chunked_head(self):
        if False:
            for i in range(10):
                print('nop')
        chunked_start = 'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\na\r\nhello world\r\n1\r\nd\r\n'
        sock = FakeSocket(chunked_start + last_chunk + chunked_end)
        resp = client.HTTPResponse(sock, method='HEAD')
        resp.begin()
        self.assertEqual(resp.read(), b'')
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.reason, 'OK')
        self.assertTrue(resp.isclosed())
        self.assertFalse(resp.closed)
        resp.close()
        self.assertTrue(resp.closed)

    def test_readinto_chunked_head(self):
        if False:
            print('Hello World!')
        chunked_start = 'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\na\r\nhello world\r\n1\r\nd\r\n'
        sock = FakeSocket(chunked_start + last_chunk + chunked_end)
        resp = client.HTTPResponse(sock, method='HEAD')
        resp.begin()
        b = bytearray(5)
        n = resp.readinto(b)
        self.assertEqual(n, 0)
        self.assertEqual(bytes(b), b'\x00' * 5)
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.reason, 'OK')
        self.assertTrue(resp.isclosed())
        self.assertFalse(resp.closed)
        resp.close()
        self.assertTrue(resp.closed)

    def test_negative_content_length(self):
        if False:
            for i in range(10):
                print('nop')
        sock = FakeSocket('HTTP/1.1 200 OK\r\nContent-Length: -1\r\n\r\nHello\r\n')
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        self.assertEqual(resp.read(), b'Hello\r\n')
        self.assertTrue(resp.isclosed())

    def test_incomplete_read(self):
        if False:
            return 10
        sock = FakeSocket('HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nHello\r\n')
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        try:
            resp.read()
        except client.IncompleteRead as i:
            self.assertEqual(i.partial, b'Hello\r\n')
            self.assertEqual(repr(i), 'IncompleteRead(7 bytes read, 3 more expected)')
            self.assertEqual(str(i), 'IncompleteRead(7 bytes read, 3 more expected)')
            self.assertTrue(resp.isclosed())
        else:
            self.fail('IncompleteRead expected')

    def test_epipe(self):
        if False:
            while True:
                i = 10
        sock = EPipeSocket('HTTP/1.0 401 Authorization Required\r\nContent-type: text/html\r\nWWW-Authenticate: Basic realm="example"\r\n', b'Content-Length')
        conn = client.HTTPConnection('example.com')
        conn.sock = sock
        self.assertRaises(OSError, lambda : conn.request('PUT', '/url', 'body'))
        resp = conn.getresponse()
        self.assertEqual(401, resp.status)
        self.assertEqual('Basic realm="example"', resp.getheader('www-authenticate'))

    def test_overflowing_status_line(self):
        if False:
            print('Hello World!')
        body = 'HTTP/1.1 200 Ok' + 'k' * 65536 + '\r\n'
        resp = client.HTTPResponse(FakeSocket(body))
        self.assertRaises((client.LineTooLong, client.BadStatusLine), resp.begin)

    def test_overflowing_header_line(self):
        if False:
            print('Hello World!')
        body = 'HTTP/1.1 200 OK\r\nX-Foo: bar' + 'r' * 65536 + '\r\n\r\n'
        resp = client.HTTPResponse(FakeSocket(body))
        self.assertRaises(client.LineTooLong, resp.begin)

    def test_overflowing_header_limit_after_100(self):
        if False:
            for i in range(10):
                print('nop')
        body = 'HTTP/1.1 100 OK\r\nr\n' * 32768
        resp = client.HTTPResponse(FakeSocket(body))
        with self.assertRaises(client.HTTPException) as cm:
            resp.begin()
        self.assertIn('got more than ', str(cm.exception))
        self.assertIn('headers', str(cm.exception))

    def test_overflowing_chunked_line(self):
        if False:
            print('Hello World!')
        body = 'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n' + '0' * 65536 + 'a\r\nhello world\r\n0\r\n\r\n'
        resp = client.HTTPResponse(FakeSocket(body))
        resp.begin()
        self.assertRaises(client.LineTooLong, resp.read)

    def test_early_eof(self):
        if False:
            i = 10
            return i + 15
        body = 'HTTP/1.1 200 Ok'
        sock = FakeSocket(body)
        resp = client.HTTPResponse(sock)
        resp.begin()
        self.assertEqual(resp.read(), b'')
        self.assertTrue(resp.isclosed())
        self.assertFalse(resp.closed)
        resp.close()
        self.assertTrue(resp.closed)

    def test_error_leak(self):
        if False:
            while True:
                i = 10
        conn = client.HTTPConnection('example.com')
        response = None

        class Response(client.HTTPResponse):

            def __init__(self, *pos, **kw):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal response
                response = self
                client.HTTPResponse.__init__(self, *pos, **kw)
        conn.response_class = Response
        conn.sock = FakeSocket('Invalid status line')
        conn.request('GET', '/')
        self.assertRaises(client.BadStatusLine, conn.getresponse)
        self.assertTrue(response.closed)
        self.assertTrue(conn.sock.file_closed)

    def test_chunked_extension(self):
        if False:
            while True:
                i = 10
        extra = '3;foo=bar\r\n' + 'abc\r\n'
        expected = chunked_expected + b'abc'
        sock = FakeSocket(chunked_start + extra + last_chunk_extended + chunked_end)
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        self.assertEqual(resp.read(), expected)
        resp.close()

    def test_chunked_missing_end(self):
        if False:
            print('Hello World!')
        'some servers may serve up a short chunked encoding stream'
        expected = chunked_expected
        sock = FakeSocket(chunked_start + last_chunk)
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        self.assertEqual(resp.read(), expected)
        resp.close()

    def test_chunked_trailers(self):
        if False:
            while True:
                i = 10
        'See that trailers are read and ignored'
        expected = chunked_expected
        sock = FakeSocket(chunked_start + last_chunk + trailers + chunked_end)
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        self.assertEqual(resp.read(), expected)
        self.assertEqual(sock.file.read(), b'')
        resp.close()

    def test_chunked_sync(self):
        if False:
            while True:
                i = 10
        "Check that we don't read past the end of the chunked-encoding stream"
        expected = chunked_expected
        extradata = 'extradata'
        sock = FakeSocket(chunked_start + last_chunk + trailers + chunked_end + extradata)
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        self.assertEqual(resp.read(), expected)
        self.assertEqual(sock.file.read(), extradata.encode('ascii'))
        resp.close()

    def test_content_length_sync(self):
        if False:
            i = 10
            return i + 15
        "Check that we don't read past the end of the Content-Length stream"
        extradata = b'extradata'
        expected = b'Hello123\r\n'
        sock = FakeSocket(b'HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n' + expected + extradata)
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        self.assertEqual(resp.read(), expected)
        self.assertEqual(sock.file.read(), extradata)
        resp.close()

    def test_readlines_content_length(self):
        if False:
            for i in range(10):
                print('nop')
        extradata = b'extradata'
        expected = b'Hello123\r\n'
        sock = FakeSocket(b'HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n' + expected + extradata)
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        self.assertEqual(resp.readlines(2000), [expected])
        self.assertEqual(sock.file.read(), extradata)
        resp.close()

    def test_read1_content_length(self):
        if False:
            return 10
        extradata = b'extradata'
        expected = b'Hello123\r\n'
        sock = FakeSocket(b'HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n' + expected + extradata)
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        self.assertEqual(resp.read1(2000), expected)
        self.assertEqual(sock.file.read(), extradata)
        resp.close()

    def test_readline_bound_content_length(self):
        if False:
            for i in range(10):
                print('nop')
        extradata = b'extradata'
        expected = b'Hello123\r\n'
        sock = FakeSocket(b'HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n' + expected + extradata)
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        self.assertEqual(resp.readline(10), expected)
        self.assertEqual(resp.readline(10), b'')
        self.assertEqual(sock.file.read(), extradata)
        resp.close()

    def test_read1_bound_content_length(self):
        if False:
            i = 10
            return i + 15
        extradata = b'extradata'
        expected = b'Hello123\r\n'
        sock = FakeSocket(b'HTTP/1.1 200 OK\r\nContent-Length: 30\r\n\r\n' + expected * 3 + extradata)
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        self.assertEqual(resp.read1(20), expected * 2)
        self.assertEqual(resp.read(), expected)
        self.assertEqual(sock.file.read(), extradata)
        resp.close()

    def test_response_fileno(self):
        if False:
            return 10
        serv = socket.create_server((HOST, 0))
        self.addCleanup(serv.close)
        result = None

        def run_server():
            if False:
                print('Hello World!')
            [conn, address] = serv.accept()
            with conn, conn.makefile('rb') as reader:
                while True:
                    line = reader.readline()
                    if not line.rstrip(b'\r\n'):
                        break
                conn.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
                nonlocal result
                result = reader.read()
        thread = threading.Thread(target=run_server)
        thread.start()
        self.addCleanup(thread.join, float(1))
        conn = client.HTTPConnection(*serv.getsockname())
        conn.request('CONNECT', 'dummy:1234')
        response = conn.getresponse()
        try:
            self.assertEqual(response.status, client.OK)
            s = socket.socket(fileno=response.fileno())
            try:
                s.sendall(b'proxied data\n')
            finally:
                s.detach()
        finally:
            response.close()
            conn.close()
        thread.join()
        self.assertEqual(result, b'proxied data\n')

    def test_putrequest_override_domain_validation(self):
        if False:
            while True:
                i = 10
        '\n        It should be possible to override the default validation\n        behavior in putrequest (bpo-38216).\n        '

        class UnsafeHTTPConnection(client.HTTPConnection):

            def _validate_path(self, url):
                if False:
                    while True:
                        i = 10
                pass
        conn = UnsafeHTTPConnection('example.com')
        conn.sock = FakeSocket('')
        conn.putrequest('GET', '/\x00')

    def test_putrequest_override_host_validation(self):
        if False:
            return 10

        class UnsafeHTTPConnection(client.HTTPConnection):

            def _validate_host(self, url):
                if False:
                    while True:
                        i = 10
                pass
        conn = UnsafeHTTPConnection('example.com\r\n')
        conn.sock = FakeSocket('')
        conn.putrequest('GET', '/', skip_host=1)

    def test_putrequest_override_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        It should be possible to override the default encoding\n        to transmit bytes in another encoding even if invalid\n        (bpo-36274).\n        '

        class UnsafeHTTPConnection(client.HTTPConnection):

            def _encode_request(self, str_url):
                if False:
                    while True:
                        i = 10
                return str_url.encode('utf-8')
        conn = UnsafeHTTPConnection('example.com')
        conn.sock = FakeSocket('')
        conn.putrequest('GET', '/☃')

class ExtendedReadTest(TestCase):
    """
    Test peek(), read1(), readline()
    """
    lines = 'HTTP/1.1 200 OK\r\n\r\nhello world!\nand now \nfor something completely different\nfoo'
    lines_expected = lines[lines.find('hello'):].encode('ascii')
    lines_chunked = 'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\na\r\nhello worl\r\n3\r\nd!\n\r\n9\r\nand now \n\r\n23\r\nfor something completely different\n\r\n3\r\nfoo\r\n0\r\n\r\n'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        sock = FakeSocket(self.lines)
        resp = client.HTTPResponse(sock, method='GET')
        resp.begin()
        resp.fp = io.BufferedReader(resp.fp)
        self.resp = resp

    def test_peek(self):
        if False:
            print('Hello World!')
        resp = self.resp
        oldpeek = resp.fp.peek

        def mypeek(n=-1):
            if False:
                for i in range(10):
                    print('nop')
            p = oldpeek(n)
            if n >= 0:
                return p[:n]
            return p[:10]
        resp.fp.peek = mypeek
        all = []
        while True:
            p = resp.peek(3)
            if p:
                self.assertGreater(len(p), 0)
                p2 = resp.peek()
                self.assertGreaterEqual(len(p2), len(p))
                self.assertTrue(p2.startswith(p))
                next = resp.read(len(p2))
                self.assertEqual(next, p2)
            else:
                next = resp.read()
                self.assertFalse(next)
            all.append(next)
            if not next:
                break
        self.assertEqual(b''.join(all), self.lines_expected)

    def test_readline(self):
        if False:
            return 10
        resp = self.resp
        self._verify_readline(self.resp.readline, self.lines_expected)

    def _verify_readline(self, readline, expected):
        if False:
            print('Hello World!')
        all = []
        while True:
            line = readline(5)
            if line and line != b'foo':
                if len(line) < 5:
                    self.assertTrue(line.endswith(b'\n'))
            all.append(line)
            if not line:
                break
        self.assertEqual(b''.join(all), expected)

    def test_read1(self):
        if False:
            return 10
        resp = self.resp

        def r():
            if False:
                while True:
                    i = 10
            res = resp.read1(4)
            self.assertLessEqual(len(res), 4)
            return res
        readliner = Readliner(r)
        self._verify_readline(readliner.readline, self.lines_expected)

    def test_read1_unbounded(self):
        if False:
            i = 10
            return i + 15
        resp = self.resp
        all = []
        while True:
            data = resp.read1()
            if not data:
                break
            all.append(data)
        self.assertEqual(b''.join(all), self.lines_expected)

    def test_read1_bounded(self):
        if False:
            return 10
        resp = self.resp
        all = []
        while True:
            data = resp.read1(10)
            if not data:
                break
            self.assertLessEqual(len(data), 10)
            all.append(data)
        self.assertEqual(b''.join(all), self.lines_expected)

    def test_read1_0(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.resp.read1(0), b'')

    def test_peek_0(self):
        if False:
            while True:
                i = 10
        p = self.resp.peek(0)
        self.assertLessEqual(0, len(p))

class ExtendedReadTestChunked(ExtendedReadTest):
    """
    Test peek(), read1(), readline() in chunked mode
    """
    lines = 'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\na\r\nhello worl\r\n3\r\nd!\n\r\n9\r\nand now \n\r\n23\r\nfor something completely different\n\r\n3\r\nfoo\r\n0\r\n\r\n'

class Readliner:
    """
    a simple readline class that uses an arbitrary read function and buffering
    """

    def __init__(self, readfunc):
        if False:
            for i in range(10):
                print('nop')
        self.readfunc = readfunc
        self.remainder = b''

    def readline(self, limit):
        if False:
            i = 10
            return i + 15
        data = []
        datalen = 0
        read = self.remainder
        try:
            while True:
                idx = read.find(b'\n')
                if idx != -1:
                    break
                if datalen + len(read) >= limit:
                    idx = limit - datalen - 1
                data.append(read)
                read = self.readfunc()
                if not read:
                    idx = 0
                    break
            idx += 1
            data.append(read[:idx])
            self.remainder = read[idx:]
            return b''.join(data)
        except:
            self.remainder = b''.join(data)
            raise

class OfflineTest(TestCase):

    def test_all(self):
        if False:
            while True:
                i = 10
        expected = {'responses'}
        blacklist = {'HTTPMessage', 'parse_headers'}
        for name in dir(client):
            if name.startswith('_') or name in blacklist:
                continue
            module_object = getattr(client, name)
            if getattr(module_object, '__module__', None) == 'http.client':
                expected.add(name)
        self.assertCountEqual(client.__all__, expected)

    def test_responses(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(client.responses[client.NOT_FOUND], 'Not Found')

    def test_client_constants(self):
        if False:
            print('Hello World!')
        expected = ['CONTINUE', 'SWITCHING_PROTOCOLS', 'PROCESSING', 'OK', 'CREATED', 'ACCEPTED', 'NON_AUTHORITATIVE_INFORMATION', 'NO_CONTENT', 'RESET_CONTENT', 'PARTIAL_CONTENT', 'MULTI_STATUS', 'IM_USED', 'MULTIPLE_CHOICES', 'MOVED_PERMANENTLY', 'FOUND', 'SEE_OTHER', 'NOT_MODIFIED', 'USE_PROXY', 'TEMPORARY_REDIRECT', 'BAD_REQUEST', 'UNAUTHORIZED', 'PAYMENT_REQUIRED', 'FORBIDDEN', 'NOT_FOUND', 'METHOD_NOT_ALLOWED', 'NOT_ACCEPTABLE', 'PROXY_AUTHENTICATION_REQUIRED', 'REQUEST_TIMEOUT', 'CONFLICT', 'GONE', 'LENGTH_REQUIRED', 'PRECONDITION_FAILED', 'REQUEST_ENTITY_TOO_LARGE', 'REQUEST_URI_TOO_LONG', 'UNSUPPORTED_MEDIA_TYPE', 'REQUESTED_RANGE_NOT_SATISFIABLE', 'EXPECTATION_FAILED', 'IM_A_TEAPOT', 'MISDIRECTED_REQUEST', 'UNPROCESSABLE_ENTITY', 'LOCKED', 'FAILED_DEPENDENCY', 'UPGRADE_REQUIRED', 'PRECONDITION_REQUIRED', 'TOO_MANY_REQUESTS', 'REQUEST_HEADER_FIELDS_TOO_LARGE', 'UNAVAILABLE_FOR_LEGAL_REASONS', 'INTERNAL_SERVER_ERROR', 'NOT_IMPLEMENTED', 'BAD_GATEWAY', 'SERVICE_UNAVAILABLE', 'GATEWAY_TIMEOUT', 'HTTP_VERSION_NOT_SUPPORTED', 'INSUFFICIENT_STORAGE', 'NOT_EXTENDED', 'NETWORK_AUTHENTICATION_REQUIRED', 'EARLY_HINTS', 'TOO_EARLY']
        for const in expected:
            with self.subTest(constant=const):
                self.assertTrue(hasattr(client, const))

class SourceAddressTest(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.port = socket_helper.bind_port(self.serv)
        self.source_port = socket_helper.find_unused_port()
        self.serv.listen()
        self.conn = None

    def tearDown(self):
        if False:
            return 10
        if self.conn:
            self.conn.close()
            self.conn = None
        self.serv.close()
        self.serv = None

    def testHTTPConnectionSourceAddress(self):
        if False:
            i = 10
            return i + 15
        self.conn = client.HTTPConnection(HOST, self.port, source_address=('', self.source_port))
        self.conn.connect()
        self.assertEqual(self.conn.sock.getsockname()[1], self.source_port)

    @unittest.skipIf(not hasattr(client, 'HTTPSConnection'), 'http.client.HTTPSConnection not defined')
    def testHTTPSConnectionSourceAddress(self):
        if False:
            i = 10
            return i + 15
        self.conn = client.HTTPSConnection(HOST, self.port, source_address=('', self.source_port))

class TimeoutTest(TestCase):
    PORT = None

    def setUp(self):
        if False:
            print('Hello World!')
        self.serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        TimeoutTest.PORT = socket_helper.bind_port(self.serv)
        self.serv.listen()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.serv.close()
        self.serv = None

    def testTimeoutAttribute(self):
        if False:
            while True:
                i = 10
        self.assertIsNone(socket.getdefaulttimeout())
        socket.setdefaulttimeout(30)
        try:
            httpConn = client.HTTPConnection(HOST, TimeoutTest.PORT)
            httpConn.connect()
        finally:
            socket.setdefaulttimeout(None)
        self.assertEqual(httpConn.sock.gettimeout(), 30)
        httpConn.close()
        self.assertIsNone(socket.getdefaulttimeout())
        socket.setdefaulttimeout(30)
        try:
            httpConn = client.HTTPConnection(HOST, TimeoutTest.PORT, timeout=None)
            httpConn.connect()
        finally:
            socket.setdefaulttimeout(None)
        self.assertEqual(httpConn.sock.gettimeout(), None)
        httpConn.close()
        httpConn = client.HTTPConnection(HOST, TimeoutTest.PORT, timeout=30)
        httpConn.connect()
        self.assertEqual(httpConn.sock.gettimeout(), 30)
        httpConn.close()

class PersistenceTest(TestCase):

    def test_reuse_reconnect(self):
        if False:
            return 10
        tests = (('1.0', '', False), ('1.0', 'Connection: keep-alive\r\n', True), ('1.1', '', True), ('1.1', 'Connection: close\r\n', False), ('1.0', 'Connection: keep-ALIVE\r\n', True), ('1.1', 'Connection: cloSE\r\n', False))
        for (version, header, reuse) in tests:
            with self.subTest(version=version, header=header):
                msg = 'HTTP/{} 200 OK\r\n{}Content-Length: 12\r\n\r\nDummy body\r\n'.format(version, header)
                conn = FakeSocketHTTPConnection(msg)
                self.assertIsNone(conn.sock)
                conn.request('GET', '/open-connection')
                with conn.getresponse() as response:
                    self.assertEqual(conn.sock is None, not reuse)
                    response.read()
                self.assertEqual(conn.sock is None, not reuse)
                self.assertEqual(conn.connections, 1)
                conn.request('GET', '/subsequent-request')
                self.assertEqual(conn.connections, 1 if reuse else 2)

    def test_disconnected(self):
        if False:
            return 10

        def make_reset_reader(text):
            if False:
                print('Hello World!')
            'Return BufferedReader that raises ECONNRESET at EOF'
            stream = io.BytesIO(text)

            def readinto(buffer):
                if False:
                    print('Hello World!')
                size = io.BytesIO.readinto(stream, buffer)
                if size == 0:
                    raise ConnectionResetError()
                return size
            stream.readinto = readinto
            return io.BufferedReader(stream)
        tests = ((io.BytesIO, client.RemoteDisconnected), (make_reset_reader, ConnectionResetError))
        for (stream_factory, exception) in tests:
            with self.subTest(exception=exception):
                conn = FakeSocketHTTPConnection(b'', stream_factory)
                conn.request('GET', '/eof-response')
                self.assertRaises(exception, conn.getresponse)
                self.assertIsNone(conn.sock)
                conn.request('GET', '/reconnect')
                self.assertEqual(conn.connections, 2)

    def test_100_close(self):
        if False:
            i = 10
            return i + 15
        conn = FakeSocketHTTPConnection(b'HTTP/1.1 100 Continue\r\n\r\n')
        conn.request('GET', '/', headers={'Expect': '100-continue'})
        self.assertRaises(client.RemoteDisconnected, conn.getresponse)
        self.assertIsNone(conn.sock)
        conn.request('GET', '/reconnect')
        self.assertEqual(conn.connections, 2)

class HTTPSTest(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(client, 'HTTPSConnection'):
            self.skipTest('ssl support required')

    def make_server(self, certfile):
        if False:
            print('Hello World!')
        from test.ssl_servers import make_https_server
        return make_https_server(self, certfile=certfile)

    def test_attributes(self):
        if False:
            i = 10
            return i + 15
        h = client.HTTPSConnection(HOST, TimeoutTest.PORT, timeout=30)
        self.assertEqual(h.timeout, 30)

    def test_networked(self):
        if False:
            for i in range(10):
                print('nop')
        import ssl
        support.requires('network')
        with socket_helper.transient_internet('self-signed.pythontest.net'):
            h = client.HTTPSConnection('self-signed.pythontest.net', 443)
            with self.assertRaises(ssl.SSLError) as exc_info:
                h.request('GET', '/')
            self.assertEqual(exc_info.exception.reason, 'CERTIFICATE_VERIFY_FAILED')

    def test_networked_noverification(self):
        if False:
            for i in range(10):
                print('nop')
        import ssl
        support.requires('network')
        with socket_helper.transient_internet('self-signed.pythontest.net'):
            context = ssl._create_unverified_context()
            h = client.HTTPSConnection('self-signed.pythontest.net', 443, context=context)
            h.request('GET', '/')
            resp = h.getresponse()
            h.close()
            self.assertIn('nginx', resp.getheader('server'))
            resp.close()

    @support.system_must_validate_cert
    def test_networked_trusted_by_default_cert(self):
        if False:
            while True:
                i = 10
        support.requires('network')
        with socket_helper.transient_internet('www.python.org'):
            h = client.HTTPSConnection('www.python.org', 443)
            h.request('GET', '/')
            resp = h.getresponse()
            content_type = resp.getheader('content-type')
            resp.close()
            h.close()
            self.assertIn('text/html', content_type)

    def test_networked_good_cert(self):
        if False:
            while True:
                i = 10
        import ssl
        support.requires('network')
        selfsigned_pythontestdotnet = 'self-signed.pythontest.net'
        with socket_helper.transient_internet(selfsigned_pythontestdotnet):
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertEqual(context.check_hostname, True)
            context.load_verify_locations(CERT_selfsigned_pythontestdotnet)
            try:
                h = client.HTTPSConnection(selfsigned_pythontestdotnet, 443, context=context)
                h.request('GET', '/')
                resp = h.getresponse()
            except ssl.SSLError as ssl_err:
                ssl_err_str = str(ssl_err)
                if re.search('(?i)key.too.weak', ssl_err_str):
                    raise unittest.SkipTest(f'Got {ssl_err_str} trying to connect to {selfsigned_pythontestdotnet}. See https://bugs.python.org/issue36816.')
                raise
            server_string = resp.getheader('server')
            resp.close()
            h.close()
            self.assertIn('nginx', server_string)

    def test_networked_bad_cert(self):
        if False:
            print('Hello World!')
        import ssl
        support.requires('network')
        with socket_helper.transient_internet('self-signed.pythontest.net'):
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.load_verify_locations(CERT_localhost)
            h = client.HTTPSConnection('self-signed.pythontest.net', 443, context=context)
            with self.assertRaises(ssl.SSLError) as exc_info:
                h.request('GET', '/')
            self.assertEqual(exc_info.exception.reason, 'CERTIFICATE_VERIFY_FAILED')

    def test_local_unknown_cert(self):
        if False:
            print('Hello World!')
        import ssl
        server = self.make_server(CERT_localhost)
        h = client.HTTPSConnection('localhost', server.port)
        with self.assertRaises(ssl.SSLError) as exc_info:
            h.request('GET', '/')
        self.assertEqual(exc_info.exception.reason, 'CERTIFICATE_VERIFY_FAILED')

    def test_local_good_hostname(self):
        if False:
            return 10
        import ssl
        server = self.make_server(CERT_localhost)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(CERT_localhost)
        h = client.HTTPSConnection('localhost', server.port, context=context)
        self.addCleanup(h.close)
        h.request('GET', '/nonexistent')
        resp = h.getresponse()
        self.addCleanup(resp.close)
        self.assertEqual(resp.status, 404)

    def test_local_bad_hostname(self):
        if False:
            return 10
        import ssl
        server = self.make_server(CERT_fakehostname)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(CERT_fakehostname)
        h = client.HTTPSConnection('localhost', server.port, context=context)
        with self.assertRaises(ssl.CertificateError):
            h.request('GET', '/')
        with support.check_warnings(('', DeprecationWarning)):
            h = client.HTTPSConnection('localhost', server.port, context=context, check_hostname=True)
        with self.assertRaises(ssl.CertificateError):
            h.request('GET', '/')
        context.check_hostname = False
        with support.check_warnings(('', DeprecationWarning)):
            h = client.HTTPSConnection('localhost', server.port, context=context, check_hostname=False)
        h.request('GET', '/nonexistent')
        resp = h.getresponse()
        resp.close()
        h.close()
        self.assertEqual(resp.status, 404)
        context.check_hostname = False
        h = client.HTTPSConnection('localhost', server.port, context=context)
        h.request('GET', '/nonexistent')
        resp = h.getresponse()
        self.assertEqual(resp.status, 404)
        resp.close()
        h.close()
        with support.check_warnings(('', DeprecationWarning)):
            h = client.HTTPSConnection('localhost', server.port, context=context, check_hostname=True)
        with self.assertRaises(ssl.CertificateError):
            h.request('GET', '/')

    @unittest.skipIf(not hasattr(client, 'HTTPSConnection'), 'http.client.HTTPSConnection not available')
    def test_host_port(self):
        if False:
            return 10
        for hp in ('www.python.org:abc', 'user:password@www.python.org'):
            self.assertRaises(client.InvalidURL, client.HTTPSConnection, hp)
        for (hp, h, p) in (('[fe80::207:e9ff:fe9b]:8000', 'fe80::207:e9ff:fe9b', 8000), ('www.python.org:443', 'www.python.org', 443), ('www.python.org:', 'www.python.org', 443), ('www.python.org', 'www.python.org', 443), ('[fe80::207:e9ff:fe9b]', 'fe80::207:e9ff:fe9b', 443), ('[fe80::207:e9ff:fe9b]:', 'fe80::207:e9ff:fe9b', 443)):
            c = client.HTTPSConnection(hp)
            self.assertEqual(h, c.host)
            self.assertEqual(p, c.port)

    def test_tls13_pha(self):
        if False:
            i = 10
            return i + 15
        import ssl
        if not ssl.HAS_TLSv1_3:
            self.skipTest('TLS 1.3 support required')
        h = client.HTTPSConnection('localhost', 443)
        self.assertTrue(h._context.post_handshake_auth)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.assertFalse(context.post_handshake_auth)
        h = client.HTTPSConnection('localhost', 443, context=context)
        self.assertIs(h._context, context)
        self.assertFalse(h._context.post_handshake_auth)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'key_file, cert_file and check_hostname are deprecated', DeprecationWarning)
            h = client.HTTPSConnection('localhost', 443, context=context, cert_file=CERT_localhost)
        self.assertTrue(h._context.post_handshake_auth)

class RequestBodyTest(TestCase):
    """Test cases where a request includes a message body."""

    def setUp(self):
        if False:
            return 10
        self.conn = client.HTTPConnection('example.com')
        self.conn.sock = self.sock = FakeSocket('')
        self.conn.sock = self.sock

    def get_headers_and_fp(self):
        if False:
            print('Hello World!')
        f = io.BytesIO(self.sock.data)
        f.readline()
        message = client.parse_headers(f)
        return (message, f)

    def test_list_body(self):
        if False:
            return 10
        cases = (([b'foo', b'bar'], b'3\r\nfoo\r\n3\r\nbar\r\n0\r\n\r\n'), ((b'foo', b'bar'), b'3\r\nfoo\r\n3\r\nbar\r\n0\r\n\r\n'))
        for (body, expected) in cases:
            with self.subTest(body):
                self.conn = client.HTTPConnection('example.com')
                self.conn.sock = self.sock = FakeSocket('')
                self.conn.request('PUT', '/url', body)
                (msg, f) = self.get_headers_and_fp()
                self.assertNotIn('Content-Type', msg)
                self.assertNotIn('Content-Length', msg)
                self.assertEqual(msg.get('Transfer-Encoding'), 'chunked')
                self.assertEqual(expected, f.read())

    def test_manual_content_length(self):
        if False:
            print('Hello World!')
        self.conn.request('PUT', '/url', 'body', {'Content-Length': '42'})
        (message, f) = self.get_headers_and_fp()
        self.assertEqual('42', message.get('content-length'))
        self.assertEqual(4, len(f.read()))

    def test_ascii_body(self):
        if False:
            for i in range(10):
                print('nop')
        self.conn.request('PUT', '/url', 'body')
        (message, f) = self.get_headers_and_fp()
        self.assertEqual('text/plain', message.get_content_type())
        self.assertIsNone(message.get_charset())
        self.assertEqual('4', message.get('content-length'))
        self.assertEqual(b'body', f.read())

    def test_latin1_body(self):
        if False:
            print('Hello World!')
        self.conn.request('PUT', '/url', 'bodyÁ')
        (message, f) = self.get_headers_and_fp()
        self.assertEqual('text/plain', message.get_content_type())
        self.assertIsNone(message.get_charset())
        self.assertEqual('5', message.get('content-length'))
        self.assertEqual(b'body\xc1', f.read())

    def test_bytes_body(self):
        if False:
            return 10
        self.conn.request('PUT', '/url', b'body\xc1')
        (message, f) = self.get_headers_and_fp()
        self.assertEqual('text/plain', message.get_content_type())
        self.assertIsNone(message.get_charset())
        self.assertEqual('5', message.get('content-length'))
        self.assertEqual(b'body\xc1', f.read())

    def test_text_file_body(self):
        if False:
            while True:
                i = 10
        self.addCleanup(support.unlink, support.TESTFN)
        with open(support.TESTFN, 'w') as f:
            f.write('body')
        with open(support.TESTFN) as f:
            self.conn.request('PUT', '/url', f)
            (message, f) = self.get_headers_and_fp()
            self.assertEqual('text/plain', message.get_content_type())
            self.assertIsNone(message.get_charset())
            self.assertIsNone(message.get('content-length'))
            self.assertEqual('chunked', message.get('transfer-encoding'))
            self.assertEqual(b'4\r\nbody\r\n0\r\n\r\n', f.read())

    def test_binary_file_body(self):
        if False:
            for i in range(10):
                print('nop')
        self.addCleanup(support.unlink, support.TESTFN)
        with open(support.TESTFN, 'wb') as f:
            f.write(b'body\xc1')
        with open(support.TESTFN, 'rb') as f:
            self.conn.request('PUT', '/url', f)
            (message, f) = self.get_headers_and_fp()
            self.assertEqual('text/plain', message.get_content_type())
            self.assertIsNone(message.get_charset())
            self.assertEqual('chunked', message.get('Transfer-Encoding'))
            self.assertNotIn('Content-Length', message)
            self.assertEqual(b'5\r\nbody\xc1\r\n0\r\n\r\n', f.read())

class HTTPResponseTest(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        body = 'HTTP/1.1 200 Ok\r\nMy-Header: first-value\r\nMy-Header:                 second-value\r\n\r\nText'
        sock = FakeSocket(body)
        self.resp = client.HTTPResponse(sock)
        self.resp.begin()

    def test_getting_header(self):
        if False:
            print('Hello World!')
        header = self.resp.getheader('My-Header')
        self.assertEqual(header, 'first-value, second-value')
        header = self.resp.getheader('My-Header', 'some default')
        self.assertEqual(header, 'first-value, second-value')

    def test_getting_nonexistent_header_with_string_default(self):
        if False:
            i = 10
            return i + 15
        header = self.resp.getheader('No-Such-Header', 'default-value')
        self.assertEqual(header, 'default-value')

    def test_getting_nonexistent_header_with_iterable_default(self):
        if False:
            for i in range(10):
                print('nop')
        header = self.resp.getheader('No-Such-Header', ['default', 'values'])
        self.assertEqual(header, 'default, values')
        header = self.resp.getheader('No-Such-Header', ('default', 'values'))
        self.assertEqual(header, 'default, values')

    def test_getting_nonexistent_header_without_default(self):
        if False:
            print('Hello World!')
        header = self.resp.getheader('No-Such-Header')
        self.assertEqual(header, None)

    def test_getting_header_defaultint(self):
        if False:
            i = 10
            return i + 15
        header = self.resp.getheader('No-Such-Header', default=42)
        self.assertEqual(header, 42)

class TunnelTests(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        response_text = 'HTTP/1.0 200 OK\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 42\r\n\r\n'
        self.host = 'proxy.com'
        self.conn = client.HTTPConnection(self.host)
        self.conn._create_connection = self._create_connection(response_text)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.conn.close()

    def _create_connection(self, response_text):
        if False:
            for i in range(10):
                print('nop')

        def create_connection(address, timeout=None, source_address=None):
            if False:
                print('Hello World!')
            return FakeSocket(response_text, host=address[0], port=address[1])
        return create_connection

    def test_set_tunnel_host_port_headers(self):
        if False:
            i = 10
            return i + 15
        tunnel_host = 'destination.com'
        tunnel_port = 8888
        tunnel_headers = {'User-Agent': 'Mozilla/5.0 (compatible, MSIE 11)'}
        self.conn.set_tunnel(tunnel_host, port=tunnel_port, headers=tunnel_headers)
        self.conn.request('HEAD', '/', '')
        self.assertEqual(self.conn.sock.host, self.host)
        self.assertEqual(self.conn.sock.port, client.HTTP_PORT)
        self.assertEqual(self.conn._tunnel_host, tunnel_host)
        self.assertEqual(self.conn._tunnel_port, tunnel_port)
        self.assertEqual(self.conn._tunnel_headers, tunnel_headers)

    def test_disallow_set_tunnel_after_connect(self):
        if False:
            i = 10
            return i + 15
        self.conn.connect()
        self.assertRaises(RuntimeError, self.conn.set_tunnel, 'destination.com')

    def test_connect_with_tunnel(self):
        if False:
            for i in range(10):
                print('nop')
        self.conn.set_tunnel('destination.com')
        self.conn.request('HEAD', '/', '')
        self.assertEqual(self.conn.sock.host, self.host)
        self.assertEqual(self.conn.sock.port, client.HTTP_PORT)
        self.assertIn(b'CONNECT destination.com', self.conn.sock.data)
        self.assertNotIn(b'Host: destination.com:None', self.conn.sock.data)
        self.assertIn(b'Host: destination.com', self.conn.sock.data)
        self.assertNotIn(b'Host: proxy.com', self.conn.sock.data)

    def test_tunnel_connect_single_send_connection_setup(self):
        if False:
            print('Hello World!')
        'Regresstion test for https://bugs.python.org/issue43332.'
        with mock.patch.object(self.conn, 'send') as mock_send:
            self.conn.set_tunnel('destination.com')
            self.conn.connect()
            self.conn.request('GET', '/')
        mock_send.assert_called()
        self.assertGreater(len(mock_send.mock_calls), 1, msg=f'unexpected number of send calls: {mock_send.mock_calls}')
        proxy_setup_data_sent = mock_send.mock_calls[0][1][0]
        self.assertIn(b'CONNECT destination.com', proxy_setup_data_sent)
        self.assertTrue(proxy_setup_data_sent.endswith(b'\r\n\r\n'), msg=f'unexpected proxy data sent {proxy_setup_data_sent!r}')

    def test_connect_put_request(self):
        if False:
            while True:
                i = 10
        self.conn.set_tunnel('destination.com')
        self.conn.request('PUT', '/', '')
        self.assertEqual(self.conn.sock.host, self.host)
        self.assertEqual(self.conn.sock.port, client.HTTP_PORT)
        self.assertIn(b'CONNECT destination.com', self.conn.sock.data)
        self.assertIn(b'Host: destination.com', self.conn.sock.data)

    def test_tunnel_debuglog(self):
        if False:
            while True:
                i = 10
        expected_header = 'X-Dummy: 1'
        response_text = 'HTTP/1.0 200 OK\r\n{}\r\n\r\n'.format(expected_header)
        self.conn.set_debuglevel(1)
        self.conn._create_connection = self._create_connection(response_text)
        self.conn.set_tunnel('destination.com')
        with support.captured_stdout() as output:
            self.conn.request('PUT', '/', '')
        lines = output.getvalue().splitlines()
        self.assertIn('header: {}'.format(expected_header), lines)
if __name__ == '__main__':
    unittest.main(verbosity=2)