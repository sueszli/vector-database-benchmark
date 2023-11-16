__doc__ = 'Tiny HTTP Proxy.\n\nThis module implements GET, HEAD, POST, PUT and DELETE methods\non BaseHTTPServer, and behaves as an HTTP proxy.  The CONNECT\nmethod is also implemented experimentally, but has not been\ntested yet.\n\nAny help will be greatly appreciated.\t\tSUZUKI Hisao\n\nPorted to Python 3 and modified for sslyze by @nabla_c0d3.\n'
__version__ = '0.3.0'
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, urlunparse
from socketserver import ThreadingMixIn
import select
import logging
import socket

class ProxyHandler(BaseHTTPRequestHandler):
    __base = BaseHTTPRequestHandler
    __base_handle = __base.handle
    server_version = 'TinyHTTPProxy/' + __version__
    rbufsize = 0

    def handle(self):
        if False:
            for i in range(10):
                print('nop')
        (ip, port) = self.client_address
        if hasattr(self, 'allowed_clients') and ip not in self.allowed_clients:
            self.raw_requestline = self.rfile.readline()
            if self.parse_request():
                self.send_error(403)
        else:
            self.__base_handle()

    def _connect_to(self, netloc, soc):
        if False:
            for i in range(10):
                print('nop')
        i = netloc.find(':')
        if i >= 0:
            host_port = (netloc[:i], int(netloc[i + 1:]))
        else:
            host_port = (netloc, 80)
        logging.warning('Connecting to {}'.format(host_port))
        try:
            soc.connect(host_port)
        except socket.error as arg:
            try:
                msg = arg[1]
            except Exception:
                msg = arg
            self.send_error(404, msg)
            return 0
        return 1
    RESPONSE_FORMAT = '{protocol} 200 Connection established\r\nProxy-agent: {version}\r\n\r\n'

    def do_CONNECT(self):
        if False:
            print('Hello World!')
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if self._connect_to(self.path, soc):
                self.log_request(200)
                response = self.RESPONSE_FORMAT.format(protocol=self.protocol_version, version=self.version_string())
                self.wfile.write(response.encode('ascii'))
                self._read_write(soc, 300)
        finally:
            logging.warning('Finished do_CONNECT()')
            soc.close()
            self.connection.close()

    def do_GET(self):
        if False:
            i = 10
            return i + 15
        (scm, netloc, path, params, query, fragment) = urlparse(self.path, 'http')
        if scm != 'http' or fragment or (not netloc):
            self.send_error(400, 'bad url %s' % self.path)
            return
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if self._connect_to(netloc, soc):
                self.log_request()
                soc.send('%s %s %s\r\n' % (self.command, urlunparse(('', '', path, params, query, '')), self.request_version))
                self.headers['Connection'] = 'close'
                del self.headers['Proxy-Connection']
                for key_val in self.headers.items():
                    soc.send('%s: %s\r\n' % key_val)
                soc.send('\r\n')
                self._read_write(soc)
        finally:
            logging.warning('Finished do_GET()')
            soc.close()
            self.connection.close()

    def _read_write(self, soc, max_idling=20):
        if False:
            for i in range(10):
                print('nop')
        iw = [self.connection, soc]
        ow = []
        count = 0
        while 1:
            count += 1
            (ins, _, exs) = select.select(iw, ow, iw, 3)
            if exs:
                break
            if ins:
                for i in ins:
                    if i is soc:
                        out = self.connection
                    else:
                        out = soc
                    data = i.recv(8192)
                    if data:
                        out.send(data)
                        count = 0
            else:
                logging.warning('Idle')
            if count == max_idling:
                break
    do_HEAD = do_GET
    do_POST = do_GET
    do_PUT = do_GET
    do_DELETE = do_GET

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass