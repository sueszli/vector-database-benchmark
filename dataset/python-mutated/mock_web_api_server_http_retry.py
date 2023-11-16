import logging
import sys
import threading
import time
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from multiprocessing.context import Process
from typing import Type
from unittest import TestCase
from tests.helpers import get_mock_server_mode

class MockHandler(SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    default_request_version = 'HTTP/1.1'
    logger = logging.getLogger(__name__)
    state = {'request_count': 0}

    def set_common_headers(self):
        if False:
            while True:
                i = 10
        self.send_header('content-type', 'application/json;charset=utf-8')
        self.send_header('connection', 'close')
        self.end_headers()
    success_response = {'ok': True}

    def _handle(self):
        if False:
            return 10
        self.state['request_count'] += 1
        try:
            header = self.headers['authorization']
            pattern = str(header).split('xoxb-', 1)[1]
            if self.state['request_count'] % 2 == 1:
                if 'remote_disconnected' in pattern:
                    self.finish()
                    return
            if pattern.isnumeric():
                self.send_response(int(pattern))
                self.set_common_headers()
                self.wfile.write('{"ok":false}'.encode('utf-8'))
                return
            if pattern == 'ratelimited':
                self.send_response(429)
                self.send_header('retry-after', 1)
                self.set_common_headers()
                self.wfile.write('{"ok":false,"error":"ratelimited"}'.encode('utf-8'))
                self.wfile.close()
                return
            if pattern == 'timeout':
                time.sleep(2)
                self.send_response(200)
                self.wfile.write('{"ok":true}'.encode('utf-8'))
                self.wfile.close()
                return
            self.send_response(HTTPStatus.OK)
            self.set_common_headers()
            self.wfile.write('{"ok":true}'.encode('utf-8'))
            self.wfile.close()
        except Exception as e:
            self.logger.error(str(e), exc_info=True)
            raise

    def do_GET(self):
        if False:
            while True:
                i = 10
        self._handle()

    def do_POST(self):
        if False:
            for i in range(10):
                print('nop')
        self._handle()

class MockServerProcessTarget:

    def __init__(self, handler: Type[SimpleHTTPRequestHandler]=MockHandler):
        if False:
            print('Hello World!')
        self.handler = handler

    def run(self):
        if False:
            return 10
        self.handler.state = {'request_count': 0}
        self.server = HTTPServer(('localhost', 8889), self.handler)
        try:
            self.server.serve_forever(0.05)
        finally:
            self.server.server_close()

    def stop(self):
        if False:
            while True:
                i = 10
        self.handler.state = {'request_count': 0}
        self.server.shutdown()
        self.join()

class MockServerThread(threading.Thread):

    def __init__(self, test: TestCase, handler: Type[SimpleHTTPRequestHandler]=MockHandler):
        if False:
            i = 10
            return i + 15
        threading.Thread.__init__(self)
        self.handler = handler
        self.test = test

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.server = HTTPServer(('localhost', 8889), self.handler)
        self.test.server_url = 'http://localhost:8889'
        (self.test.host, self.test.port) = self.server.socket.getsockname()
        self.test.server_started.set()
        self.test = None
        try:
            self.server.serve_forever()
        finally:
            self.server.server_close()

    def stop(self):
        if False:
            while True:
                i = 10
        self.server.shutdown()
        self.join()

def setup_mock_retry_web_api_server(test: TestCase):
    if False:
        i = 10
        return i + 15
    if get_mock_server_mode() == 'threading':
        test.server_started = threading.Event()
        test.thread = MockServerThread(test)
        test.thread.start()
        test.server_started.wait()
    else:
        target = MockServerProcessTarget()
        test.server_url = 'http://localhost:8889'
        (test.host, test.port) = ('localhost', 8889)
        test.process = Process(target=target.run, daemon=True)
        test.process.start()
        time.sleep(0.1)

def cleanup_mock_retry_web_api_server(test: TestCase):
    if False:
        for i in range(10):
            print('nop')
    if get_mock_server_mode() == 'threading':
        test.thread.stop()
        test.thread = None
    else:
        test.monitor_thread.stop()
        retry_count = 0
        while test.process.is_alive():
            test.process.terminate()
            time.sleep(0.01)
            retry_count += 1
            if retry_count >= 100:
                raise Exception('Failed to stop the mock server!')
        if sys.version_info.major == 3 and sys.version_info.minor > 6:
            test.process.close()
        test.process = None