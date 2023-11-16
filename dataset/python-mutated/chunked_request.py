import http.client
import os
import ssl
import time
from io import StringIO
from urllib.parse import urlparse, unquote
from chart_studio.api import utils

class Stream:

    def __init__(self, server, port=80, headers={}, url='/', ssl_enabled=False, ssl_verification_enabled=True):
        if False:
            return 10
        'Initialize a stream object and an HTTP or HTTPS connection\n        with chunked Transfer-Encoding to server:port with optional headers.\n        '
        self.maxtries = 5
        self._tries = 0
        self._delay = 1
        self._closed = False
        self._server = server
        self._port = port
        self._headers = headers
        self._url = url
        self._ssl_enabled = ssl_enabled
        self._ssl_verification_enabled = ssl_verification_enabled
        self._connect()

    def write(self, data, reconnect_on=('', 200, 502)):
        if False:
            print('Hello World!')
        'Send `data` to the server in chunk-encoded form.\n        Check the connection before writing and reconnect\n        if disconnected and if the response status code is in `reconnect_on`.\n\n        The response may either be an HTTPResponse object or an empty string.\n        '
        if not self._isconnected():
            response = self._getresponse()
            if response == '' and '' in reconnect_on or (response and isinstance(response, http.client.HTTPResponse) and (response.status in reconnect_on)):
                self._reconnect()
            elif response and isinstance(response, http.client.HTTPResponse):
                raise Exception('Server responded with status code: {status_code}\nand message: {msg}.'.format(status_code=response.status, msg=response.read()))
            elif response == '':
                raise Exception('Attempted to write but socket was not connected.')
        try:
            msg = data
            msglen = format(len(msg), 'x')
            self._conn.sock.setblocking(1)
            self._conn.send('{msglen}\r\n{msg}\r\n'.format(msglen=msglen, msg=msg).encode('utf-8'))
            self._conn.sock.setblocking(0)
        except http.client.socket.error:
            self._reconnect()
            self.write(data)

    def _get_proxy_config(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine if self._url should be passed through a proxy. If so, return\n        the appropriate proxy_server and proxy_port. Assumes https_proxy is used\n        when ssl_enabled=True.\n\n        '
        proxy_server = None
        proxy_port = None
        proxy_username = None
        proxy_password = None
        proxy_auth = None
        ssl_enabled = self._ssl_enabled
        if ssl_enabled:
            proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        else:
            proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        no_proxy = os.environ.get('no_proxy') or os.environ.get('NO_PROXY')
        no_proxy_url = no_proxy and self._server in no_proxy
        if proxy and (not no_proxy_url):
            p = urlparse(proxy)
            proxy_server = p.hostname
            proxy_port = p.port
            proxy_username = p.username
            proxy_password = p.password
        if proxy_username and proxy_password:
            username = unquote(proxy_username)
            password = unquote(proxy_password)
            proxy_auth = utils.basic_auth(username, password)
        return (proxy_server, proxy_port, proxy_auth)

    def _get_ssl_context(self):
        if False:
            return 10
        '\n        Return an unverified context if ssl verification is disabled.\n\n        '
        context = None
        if not self._ssl_verification_enabled:
            context = ssl._create_unverified_context()
        return context

    def _connect(self):
        if False:
            i = 10
            return i + 15
        'Initialize an HTTP/HTTPS connection with chunked Transfer-Encoding\n        to server:port with optional headers.\n        '
        server = self._server
        port = self._port
        headers = self._headers
        ssl_enabled = self._ssl_enabled
        (proxy_server, proxy_port, proxy_auth) = self._get_proxy_config()
        if proxy_server and proxy_port:
            if ssl_enabled:
                context = self._get_ssl_context()
                self._conn = http.client.HTTPSConnection(proxy_server, proxy_port, context=context)
            else:
                self._conn = http.client.HTTPConnection(proxy_server, proxy_port)
            tunnel_headers = None
            if proxy_auth:
                tunnel_headers = {'Proxy-Authorization': proxy_auth}
            self._conn.set_tunnel(server, port, headers=tunnel_headers)
        elif ssl_enabled:
            context = self._get_ssl_context()
            self._conn = http.client.HTTPSConnection(server, port, context=context)
        else:
            self._conn = http.client.HTTPConnection(server, port)
        self._conn.putrequest('POST', self._url)
        self._conn.putheader('Transfer-Encoding', 'chunked')
        for header in headers:
            self._conn.putheader(header, headers[header])
        self._conn.endheaders()
        self._conn.sock.setblocking(False)
        self._bytes = b''
        self._reset_retries()
        time.sleep(0.5)

    def close(self):
        if False:
            i = 10
            return i + 15
        'Close the connection to server.\n\n        If available, return a http.client.HTTPResponse object.\n\n        Closing the connection involves sending the\n        Transfer-Encoding terminating bytes.\n        '
        self._reset_retries()
        self._closed = True
        try:
            self._conn.send('\r\n0\r\n\r\n'.encode('utf-8'))
        except http.client.socket.error:
            return ''
        return self._getresponse()

    def _getresponse(self):
        if False:
            i = 10
            return i + 15
        "Read from recv and return a HTTPResponse object if possible.\n        Either\n        1 - The client has succesfully closed the connection: Return ''\n        2 - The server has already closed the connection: Return the response\n            if possible.\n        "
        self._conn.sock.setblocking(True)
        response = self._bytes
        while True:
            try:
                _bytes = self._conn.sock.recv(1)
            except http.client.socket.error:
                return b''
            if _bytes == b'':
                break
            else:
                response += _bytes
        self._conn.sock.setblocking(False)
        if response != b'':
            try:
                response = http.client.HTTPResponse(_FakeSocket(response))
                response.begin()
            except:
                response = b''
        return response

    def _isconnected(self):
        if False:
            while True:
                i = 10
        'Return True if the socket is still connected\n        to the server, False otherwise.\n\n        This check is done in 3 steps:\n        1 - Check if we have closed the connection\n        2 - Check if the original socket connection failed\n        3 - Check if the server has returned any data. If they have,\n            assume that the server closed the response after they sent\n            the data, i.e. that the data was the HTTP response.\n        '
        if self._closed:
            return False
        if self._conn.sock is None:
            return False
        try:
            self._bytes = b''
            self._bytes = self._conn.sock.recv(1)
            return False
        except http.client.socket.error as e:
            if e.errno == 35 or e.errno == 10035:
                return True
            elif e.errno == 54 or e.errno == 10054:
                return False
            elif e.errno == 11:
                return True
            elif isinstance(e, ssl.SSLError):
                if e.errno == 2:
                    return True
                raise e
            else:
                raise e

    def _reconnect(self):
        if False:
            print('Hello World!')
        'Connect if disconnected.\n        Retry self.maxtries times with delays\n        '
        if not self._isconnected():
            try:
                self._connect()
            except http.client.socket.error as e:
                if e.errno == 61 or e.errno == 10061:
                    time.sleep(self._delay)
                    self._delay += self._delay
                    self._tries += 1
                    if self._tries < self.maxtries:
                        self._reconnect()
                    else:
                        self._reset_retries()
                        raise e
                else:
                    raise e
        self._closed = False

    def _reset_retries(self):
        if False:
            print('Hello World!')
        'Reset the connect counters and delays'
        self._tries = 0
        self._delay = 1

class _FakeSocket(StringIO):

    def makefile(self, *args, **kwargs):
        if False:
            return 10
        return self