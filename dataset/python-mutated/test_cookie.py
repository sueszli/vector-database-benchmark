from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from .utils import http

class TestIntegration:

    def setup_mock_server(self, handler):
        if False:
            print('Hello World!')
        'Configure mock server.'
        self.mock_server = HTTPServer(('localhost', 0), handler)
        (_, self.mock_server_port) = self.mock_server.server_address
        self.mock_server_thread = Thread(target=self.mock_server.serve_forever)
        self.mock_server_thread.setDaemon(True)
        self.mock_server_thread.start()

    def test_cookie_parser(self):
        if False:
            i = 10
            return i + 15
        'Not directly testing HTTPie but `requests` to ensure their cookies handling\n        is still as expected by `get_expired_cookies()`.\n        '

        class MockServerRequestHandler(BaseHTTPRequestHandler):
            """"HTTP request handler."""

            def do_GET(self):
                if False:
                    for i in range(10):
                        print('nop')
                'Handle GET requests.'
                cookie = SimpleCookie()
                cookie['hello'] = 'world'
                cookie['hello']['path'] = self.path
                cookie['oatmeal_raisin'] = 'is the best'
                cookie['oatmeal_raisin']['path'] = self.path
                self.send_response(200)
                self.send_header('Set-Cookie', cookie.output())
                self.end_headers()
        self.setup_mock_server(MockServerRequestHandler)
        response = http(f'http://localhost:{self.mock_server_port}/')
        assert 'Set-Cookie: hello=world; Path=/' in response
        assert 'Set-Cookie: oatmeal_raisin="is the best"; Path=/' in response