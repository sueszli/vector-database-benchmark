""" Define a simple web server for testing purpose.

Serve html pages that are needed by the webdriver unit tests.
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.request import urlopen
import pytest
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8000
HTML_ROOT = Path(__file__).parent.parent.parent
WEBDRIVER = os.environ.get('WEBDRIVER', '<undefined>')
__all__ = ('file_server', 'HtmlOnlyHandler', 'SimpleWebServer')

class HtmlOnlyHandler(BaseHTTPRequestHandler):
    """Http handler."""

    def do_GET(self) -> None:
        if False:
            while True:
                i = 10
        'GET method handler.'
        path = self.path.split('?')[0]
        if path.startswith('/'):
            path = path[1:]
        try:
            with open(HTML_ROOT / path, mode='rb') as f:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f.read())
        except OSError:
            self.send_error(404, f'File Not Found: {path}')

    def log_message(self, format: str, *args: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Override default to avoid trashing stderr'
        pass

class SimpleWebServer:
    """A very basic web server."""

    def __init__(self, host: str=DEFAULT_HOST, port: int=DEFAULT_PORT) -> None:
        if False:
            print('Hello World!')
        self.stop_serving = False
        while True:
            try:
                self.server = HTTPServer((host, port), HtmlOnlyHandler)
                self.host = host
                self.port = port
                break
            except OSError:
                log.debug(f'port {port} is in use, trying to next one')
                port += 1
        self.thread = threading.Thread(target=self._run_web_server)

    def _run_web_server(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Runs the server loop.'
        log.debug('web server started')
        while not self.stop_serving:
            self.server.handle_request()
        self.server.server_close()

    def start(self) -> None:
        if False:
            print('Hello World!')
        'Starts the server.'
        self.thread.start()

    def stop(self) -> None:
        if False:
            i = 10
            return i + 15
        'Stops the server.'
        self.stop_serving = True
        try:
            urlopen(f'http://{self.host}:{self.port}')
        except OSError:
            pass
        log.info('Shutting down the webserver')
        self.thread.join()

    def where_is(self, path: Path) -> str:
        if False:
            i = 10
            return i + 15
        path = str(path.relative_to(HTML_ROOT)).replace('\\', '/')
        return f'http://{self.host}:{self.port}/{path}'

@pytest.fixture(scope='session')
def file_server(request: pytest.FixtureRequest) -> SimpleWebServer:
    if False:
        while True:
            i = 10
    server = SimpleWebServer()
    server.start()
    request.addfinalizer(server.stop)
    return server
_html_root_error_message = f"Can't find 'common_web' directory, try setting WEBDRIVER environment variable WEBDRIVER: {WEBDRIVER} HTML_ROOT: {HTML_ROOT}"
if not os.path.isdir(HTML_ROOT):
    log.error(_html_root_error_message)
    assert 0, _html_root_error_message