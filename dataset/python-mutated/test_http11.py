from websockets.datastructures import Headers
from websockets.exceptions import SecurityError
from websockets.http11 import *
from websockets.http11 import parse_headers
from websockets.streams import StreamReader
from .utils import GeneratorTestCase

class RequestTests(GeneratorTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.reader = StreamReader()

    def parse(self):
        if False:
            for i in range(10):
                print('nop')
        return Request.parse(self.reader.read_line)

    def test_parse(self):
        if False:
            while True:
                i = 10
        self.reader.feed_data(b'GET /chat HTTP/1.1\r\nHost: server.example.com\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nOrigin: http://example.com\r\nSec-WebSocket-Protocol: chat, superchat\r\nSec-WebSocket-Version: 13\r\n\r\n')
        request = self.assertGeneratorReturns(self.parse())
        self.assertEqual(request.path, '/chat')
        self.assertEqual(request.headers['Upgrade'], 'websocket')

    def test_parse_empty(self):
        if False:
            return 10
        self.reader.feed_eof()
        with self.assertRaises(EOFError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'connection closed while reading HTTP request line')

    def test_parse_invalid_request_line(self):
        if False:
            for i in range(10):
                print('nop')
        self.reader.feed_data(b'GET /\r\n\r\n')
        with self.assertRaises(ValueError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'invalid HTTP request line: GET /')

    def test_parse_unsupported_method(self):
        if False:
            print('Hello World!')
        self.reader.feed_data(b'OPTIONS * HTTP/1.1\r\n\r\n')
        with self.assertRaises(ValueError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'unsupported HTTP method: OPTIONS')

    def test_parse_unsupported_version(self):
        if False:
            for i in range(10):
                print('nop')
        self.reader.feed_data(b'GET /chat HTTP/1.0\r\n\r\n')
        with self.assertRaises(ValueError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'unsupported HTTP version: HTTP/1.0')

    def test_parse_invalid_header(self):
        if False:
            print('Hello World!')
        self.reader.feed_data(b'GET /chat HTTP/1.1\r\nOops\r\n')
        with self.assertRaises(ValueError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'invalid HTTP header line: Oops')

    def test_parse_body(self):
        if False:
            print('Hello World!')
        self.reader.feed_data(b'GET / HTTP/1.1\r\nContent-Length: 3\r\n\r\nYo\n')
        with self.assertRaises(ValueError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'unsupported request body')

    def test_parse_body_with_transfer_encoding(self):
        if False:
            print('Hello World!')
        self.reader.feed_data(b'GET / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n')
        with self.assertRaises(NotImplementedError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), "transfer codings aren't supported")

    def test_serialize(self):
        if False:
            for i in range(10):
                print('nop')
        request = Request('/chat', Headers([('Host', 'server.example.com'), ('Upgrade', 'websocket'), ('Connection', 'Upgrade'), ('Sec-WebSocket-Key', 'dGhlIHNhbXBsZSBub25jZQ=='), ('Origin', 'http://example.com'), ('Sec-WebSocket-Protocol', 'chat, superchat'), ('Sec-WebSocket-Version', '13')]))
        self.assertEqual(request.serialize(), b'GET /chat HTTP/1.1\r\nHost: server.example.com\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nOrigin: http://example.com\r\nSec-WebSocket-Protocol: chat, superchat\r\nSec-WebSocket-Version: 13\r\n\r\n')

class ResponseTests(GeneratorTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.reader = StreamReader()

    def parse(self):
        if False:
            print('Hello World!')
        return Response.parse(self.reader.read_line, self.reader.read_exact, self.reader.read_to_eof)

    def test_parse(self):
        if False:
            i = 10
            return i + 15
        self.reader.feed_data(b'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\nSec-WebSocket-Protocol: chat\r\n\r\n')
        response = self.assertGeneratorReturns(self.parse())
        self.assertEqual(response.status_code, 101)
        self.assertEqual(response.reason_phrase, 'Switching Protocols')
        self.assertEqual(response.headers['Upgrade'], 'websocket')
        self.assertIsNone(response.body)

    def test_parse_empty(self):
        if False:
            while True:
                i = 10
        self.reader.feed_eof()
        with self.assertRaises(EOFError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'connection closed while reading HTTP status line')

    def test_parse_invalid_status_line(self):
        if False:
            print('Hello World!')
        self.reader.feed_data(b'Hello!\r\n')
        with self.assertRaises(ValueError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'invalid HTTP status line: Hello!')

    def test_parse_unsupported_version(self):
        if False:
            for i in range(10):
                print('nop')
        self.reader.feed_data(b'HTTP/1.0 400 Bad Request\r\n\r\n')
        with self.assertRaises(ValueError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'unsupported HTTP version: HTTP/1.0')

    def test_parse_invalid_status(self):
        if False:
            for i in range(10):
                print('nop')
        self.reader.feed_data(b'HTTP/1.1 OMG WTF\r\n\r\n')
        with self.assertRaises(ValueError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'invalid HTTP status code: OMG')

    def test_parse_unsupported_status(self):
        if False:
            for i in range(10):
                print('nop')
        self.reader.feed_data(b'HTTP/1.1 007 My name is Bond\r\n\r\n')
        with self.assertRaises(ValueError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'unsupported HTTP status code: 007')

    def test_parse_invalid_reason(self):
        if False:
            return 10
        self.reader.feed_data(b'HTTP/1.1 200 \x7f\r\n\r\n')
        with self.assertRaises(ValueError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'invalid HTTP reason phrase: \x7f')

    def test_parse_invalid_header(self):
        if False:
            print('Hello World!')
        self.reader.feed_data(b'HTTP/1.1 500 Internal Server Error\r\nOops\r\n')
        with self.assertRaises(ValueError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'invalid HTTP header line: Oops')

    def test_parse_body_with_content_length(self):
        if False:
            print('Hello World!')
        self.reader.feed_data(b'HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello world!\n')
        response = self.assertGeneratorReturns(self.parse())
        self.assertEqual(response.body, b'Hello world!\n')

    def test_parse_body_without_content_length(self):
        if False:
            i = 10
            return i + 15
        self.reader.feed_data(b'HTTP/1.1 200 OK\r\n\r\nHello world!\n')
        gen = self.parse()
        self.assertGeneratorRunning(gen)
        self.reader.feed_eof()
        response = self.assertGeneratorReturns(gen)
        self.assertEqual(response.body, b'Hello world!\n')

    def test_parse_body_with_content_length_too_long(self):
        if False:
            while True:
                i = 10
        self.reader.feed_data(b'HTTP/1.1 200 OK\r\nContent-Length: 1048577\r\n\r\n')
        with self.assertRaises(SecurityError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'body too large: 1048577 bytes')

    def test_parse_body_without_content_length_too_long(self):
        if False:
            print('Hello World!')
        self.reader.feed_data(b'HTTP/1.1 200 OK\r\n\r\n' + b'a' * 1048577)
        with self.assertRaises(SecurityError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), 'body too large: over 1048576 bytes')

    def test_parse_body_with_transfer_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        self.reader.feed_data(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n')
        with self.assertRaises(NotImplementedError) as raised:
            next(self.parse())
        self.assertEqual(str(raised.exception), "transfer codings aren't supported")

    def test_parse_body_no_content(self):
        if False:
            while True:
                i = 10
        self.reader.feed_data(b'HTTP/1.1 204 No Content\r\n\r\n')
        response = self.assertGeneratorReturns(self.parse())
        self.assertIsNone(response.body)

    def test_parse_body_not_modified(self):
        if False:
            while True:
                i = 10
        self.reader.feed_data(b'HTTP/1.1 304 Not Modified\r\n\r\n')
        response = self.assertGeneratorReturns(self.parse())
        self.assertIsNone(response.body)

    def test_serialize(self):
        if False:
            i = 10
            return i + 15
        response = Response(101, 'Switching Protocols', Headers([('Upgrade', 'websocket'), ('Connection', 'Upgrade'), ('Sec-WebSocket-Accept', 's3pPLMBiTxaQ9kYGzzhZRbK+xOo='), ('Sec-WebSocket-Protocol', 'chat')]))
        self.assertEqual(response.serialize(), b'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\nSec-WebSocket-Protocol: chat\r\n\r\n')

    def test_serialize_with_body(self):
        if False:
            while True:
                i = 10
        response = Response(200, 'OK', Headers([('Content-Length', '13'), ('Content-Type', 'text/plain')]), b'Hello world!\n')
        self.assertEqual(response.serialize(), b'HTTP/1.1 200 OK\r\nContent-Length: 13\r\nContent-Type: text/plain\r\n\r\nHello world!\n')

class HeadersTests(GeneratorTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.reader = StreamReader()

    def parse_headers(self):
        if False:
            i = 10
            return i + 15
        return parse_headers(self.reader.read_line)

    def test_parse_invalid_name(self):
        if False:
            for i in range(10):
                print('nop')
        self.reader.feed_data(b'foo bar: baz qux\r\n\r\n')
        with self.assertRaises(ValueError):
            next(self.parse_headers())

    def test_parse_invalid_value(self):
        if False:
            print('Hello World!')
        self.reader.feed_data(b'foo: \x00\x00\x0f\r\n\r\n')
        with self.assertRaises(ValueError):
            next(self.parse_headers())

    def test_parse_too_long_value(self):
        if False:
            for i in range(10):
                print('nop')
        self.reader.feed_data(b'foo: bar\r\n' * 129 + b'\r\n')
        with self.assertRaises(SecurityError):
            next(self.parse_headers())

    def test_parse_too_long_line(self):
        if False:
            while True:
                i = 10
        self.reader.feed_data(b'foo: ' + b'a' * 8186 + b'\r\n\r\n')
        with self.assertRaises(SecurityError):
            next(self.parse_headers())

    def test_parse_invalid_line_ending(self):
        if False:
            while True:
                i = 10
        self.reader.feed_data(b'foo: bar\n\n')
        with self.assertRaises(EOFError):
            next(self.parse_headers())