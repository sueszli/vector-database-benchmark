import http.client
import io
import os
import urllib.error
import urllib.parse
import urllib.request
ssl = None
try:
    import ssl
except ImportError:
    pass

class Error(Exception):
    pass

class UnknownSize(Error):
    pass

class ProxyError(Error):
    pass
MIME_BOUNDARY = 'END_OF_PART'

def get_headers(http_response):
    if False:
        i = 10
        return i + 15
    'Retrieves all HTTP headers from an HTTP response from the server.\n\n    This method is provided for backwards compatibility for Python2.2 and 2.3.\n    The httplib.HTTPResponse object in 2.2 and 2.3 does not have a getheaders\n    method so this function will use getheaders if available, but if not it\n    will retrieve a few using getheader.\n    '
    if hasattr(http_response, 'getheaders'):
        return http_response.getheaders()
    else:
        headers = []
        for header in ('location', 'content-type', 'content-length', 'age', 'allow', 'cache-control', 'content-location', 'content-encoding', 'date', 'etag', 'expires', 'last-modified', 'pragma', 'server', 'set-cookie', 'transfer-encoding', 'vary', 'via', 'warning', 'www-authenticate', 'gdata-version'):
            value = http_response.getheader(header, None)
            if value is not None:
                headers.append((header, value))
        return headers

class HttpRequest(object):
    """Contains all of the parameters for an HTTP 1.1 request.

    The HTTP headers are represented by a dictionary, and it is the
    responsibility of the user to ensure that duplicate field names are combined
    into one header value according to the rules in section 4.2 of RFC 2616.
    """
    method = None
    uri = None

    def __init__(self, uri=None, method=None, headers=None):
        if False:
            i = 10
            return i + 15
        "Construct an HTTP request.\n\n        Args:\n          uri: The full path or partial path as a Uri object or a string.\n          method: The HTTP method for the request, examples include 'GET', 'POST',\n                  etc.\n          headers: dict of strings The HTTP headers to include in the request.\n        "
        self.headers = headers or {}
        self._body_parts = []
        if method is not None:
            self.method = method
        if isinstance(uri, str):
            uri = Uri.parse_uri(uri)
        self.uri = uri or Uri()

    def add_body_part(self, data, mime_type, size=None):
        if False:
            i = 10
            return i + 15
        'Adds data to the HTTP request body.\n\n        If more than one part is added, this is assumed to be a mime-multipart\n        request. This method is designed to create MIME 1.0 requests as specified\n        in RFC 1341.\n\n        Args:\n          data: str or a file-like object containing a part of the request body.\n          mime_type: str The MIME type describing the data\n          size: int Required if the data is a file like object. If the data is a\n                string, the size is calculated so this parameter is ignored.\n        '
        if hasattr(data, '__len__'):
            size = len(data)
        if size is None:
            raise UnknownSize('Each part of the body must have a known size.')
        if 'Content-Length' in self.headers:
            content_length = int(self.headers['Content-Length'])
        else:
            content_length = 0
        if len(self._body_parts) == 0:
            self.headers['Content-Type'] = mime_type
            content_length = size
            self._body_parts.append(data)
        elif len(self._body_parts) == 1:
            self._body_parts.insert(0, 'Media multipart posting')
            boundary_string = '\r\n--%s\r\n' % (MIME_BOUNDARY,)
            content_length += len(boundary_string) + size
            self._body_parts.insert(1, boundary_string)
            content_length += len('Media multipart posting')
            original_type_string = 'Content-Type: %s\r\n\r\n' % (self.headers['Content-Type'],)
            self._body_parts.insert(2, original_type_string)
            content_length += len(original_type_string)
            boundary_string = '\r\n--%s\r\n' % (MIME_BOUNDARY,)
            self._body_parts.append(boundary_string)
            content_length += len(boundary_string)
            self.headers['Content-Type'] = 'multipart/related; boundary="%s"' % (MIME_BOUNDARY,)
            self.headers['MIME-version'] = '1.0'
            type_string = 'Content-Type: %s\r\n\r\n' % mime_type
            self._body_parts.append(type_string)
            content_length += len(type_string)
            self._body_parts.append(data)
            ending_boundary_string = '\r\n--%s--' % (MIME_BOUNDARY,)
            self._body_parts.append(ending_boundary_string)
            content_length += len(ending_boundary_string)
        else:
            boundary_string = '\r\n--%s\r\n' % (MIME_BOUNDARY,)
            self._body_parts.insert(-1, boundary_string)
            content_length += len(boundary_string) + size
            type_string = 'Content-Type: %s\r\n\r\n' % mime_type
            self._body_parts.insert(-1, type_string)
            content_length += len(type_string)
            self._body_parts.insert(-1, data)
        self.headers['Content-Length'] = str(content_length)
    AddBodyPart = add_body_part

    def add_form_inputs(self, form_data, mime_type='application/x-www-form-urlencoded'):
        if False:
            print('Hello World!')
        "Form-encodes and adds data to the request body.\n\n        Args:\n          form_data: dict or sequnce or two member tuples which contains the\n                     form keys and values.\n          mime_type: str The MIME type of the form data being sent. Defaults\n                     to 'application/x-www-form-urlencoded'.\n        "
        body = urllib.parse.urlencode(form_data)
        self.add_body_part(body, bytes(mime_type, 'ascii'))
    AddFormInputs = add_form_inputs

    def _copy(self):
        if False:
            return 10
        'Creates a deep copy of this request.'
        copied_uri = Uri(self.uri.scheme, self.uri.host, self.uri.port, self.uri.path, self.uri.query.copy())
        new_request = HttpRequest(uri=copied_uri, method=self.method, headers=self.headers.copy())
        new_request._body_parts = self._body_parts[:]
        return new_request

    def _dump(self):
        if False:
            for i in range(10):
                print('nop')
        'Converts to a printable string for debugging purposes.\n\n        In order to preserve the request, it does not read from file-like objects\n        in the body.\n        '
        output = 'HTTP Request\n  method: %s\n  url: %s\n  headers:\n' % (self.method, str(self.uri))
        for (header, value) in self.headers.items():
            output += '    %s: %s\n' % (header, value)
        output += '  body sections:\n'
        i = 0
        for part in self._body_parts:
            if isinstance(part, str):
                output += '    %s: %s\n' % (i, part)
            else:
                output += '    %s: <file like object>\n' % i
            i += 1
        return output

def _apply_defaults(http_request):
    if False:
        i = 10
        return i + 15
    if http_request.uri.scheme is None:
        if http_request.uri.port == 443:
            http_request.uri.scheme = 'https'
        else:
            http_request.uri.scheme = 'http'

class Uri(object):
    """A URI as used in HTTP 1.1"""
    scheme = None
    host = None
    port = None
    path = None

    def __init__(self, scheme=None, host=None, port=None, path=None, query=None):
        if False:
            return 10
        "Constructor for a URI.\n\n        Args:\n          scheme: str This is usually 'http' or 'https'.\n          host: str The host name or IP address of the desired server.\n          post: int The server's port number.\n          path: str The path of the resource following the host. This begins with\n                a /, example: '/calendar/feeds/default/allcalendars/full'\n          query: dict of strings The URL query parameters. The keys and values are\n                 both escaped so this dict should contain the unescaped values.\n                 For example {'my key': 'val', 'second': '!!!'} will become\n                 '?my+key=val&second=%21%21%21' which is appended to the path.\n        "
        self.query = query or {}
        if scheme is not None:
            self.scheme = scheme
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port
        if path:
            self.path = path

    def _get_query_string(self):
        if False:
            return 10
        param_pairs = []
        for (key, value) in self.query.items():
            quoted_key = urllib.parse.quote_plus(str(key))
            if value is None:
                param_pairs.append(quoted_key)
            else:
                quoted_value = urllib.parse.quote_plus(str(value))
                param_pairs.append('%s=%s' % (quoted_key, quoted_value))
        return '&'.join(param_pairs)

    def _get_relative_path(self):
        if False:
            i = 10
            return i + 15
        'Returns the path with the query parameters escaped and appended.'
        param_string = self._get_query_string()
        if self.path is None:
            path = '/'
        else:
            path = self.path
        if param_string:
            return '?'.join([path, param_string])
        else:
            return path

    def _to_string(self):
        if False:
            print('Hello World!')
        if self.scheme is None and self.port == 443:
            scheme = 'https'
        elif self.scheme is None:
            scheme = 'http'
        else:
            scheme = self.scheme
        if self.path is None:
            path = '/'
        else:
            path = self.path
        if self.port is None:
            return '%s://%s%s' % (scheme, self.host, self._get_relative_path())
        else:
            return '%s://%s:%s%s' % (scheme, self.host, str(self.port), self._get_relative_path())

    def __str__(self):
        if False:
            while True:
                i = 10
        return self._to_string()

    def modify_request(self, http_request=None):
        if False:
            return 10
        'Sets HTTP request components based on the URI.'
        if http_request is None:
            http_request = HttpRequest()
        if http_request.uri is None:
            http_request.uri = Uri()
        if self.scheme:
            http_request.uri.scheme = self.scheme
        if self.port:
            http_request.uri.port = self.port
        if self.host:
            http_request.uri.host = self.host
        if self.path:
            http_request.uri.path = self.path
        if self.query:
            http_request.uri.query = self.query.copy()
        return http_request
    ModifyRequest = modify_request

    def parse_uri(uri_string):
        if False:
            print('Hello World!')
        'Creates a Uri object which corresponds to the URI string.\n\n        This method can accept partial URIs, but it will leave missing\n        members of the Uri unset.\n        '
        parts = urllib.parse.urlparse(uri_string)
        uri = Uri()
        if parts[0]:
            uri.scheme = parts[0]
        if parts[1]:
            host_parts = parts[1].split(':')
            if host_parts[0]:
                uri.host = host_parts[0]
            if len(host_parts) > 1:
                uri.port = int(host_parts[1])
        if parts[2]:
            uri.path = parts[2]
        if parts[4]:
            param_pairs = parts[4].split('&')
            for pair in param_pairs:
                pair_parts = pair.split('=')
                if len(pair_parts) > 1:
                    uri.query[urllib.parse.unquote_plus(pair_parts[0])] = urllib.parse.unquote_plus(pair_parts[1])
                elif len(pair_parts) == 1:
                    uri.query[urllib.parse.unquote_plus(pair_parts[0])] = None
        return uri
    parse_uri = staticmethod(parse_uri)
    ParseUri = parse_uri
parse_uri = Uri.parse_uri
ParseUri = Uri.parse_uri

class HttpResponse(object):
    status = None
    reason = None
    _body = None

    def __init__(self, status=None, reason=None, headers=None, body=None):
        if False:
            print('Hello World!')
        self._headers = headers or {}
        if status is not None:
            self.status = status
        if reason is not None:
            self.reason = reason
        if body is not None:
            if hasattr(body, 'read'):
                self._body = body
            else:
                self._body = io.StringIO(body)

    def getheader(self, name, default=None):
        if False:
            return 10
        if name in self._headers:
            return self._headers[name]
        else:
            return default

    def getheaders(self):
        if False:
            i = 10
            return i + 15
        return self._headers

    def read(self, amt=None):
        if False:
            for i in range(10):
                print('nop')
        if self._body is None:
            return None
        if not amt:
            return self._body.read()
        else:
            return self._body.read(amt)

def _dump_response(http_response):
    if False:
        print('Hello World!')
    'Converts to a string for printing debug messages.\n\n    Does not read the body since that may consume the content.\n    '
    output = 'HttpResponse\n  status: %s\n  reason: %s\n  headers:' % (http_response.status, http_response.reason)
    headers = get_headers(http_response)
    if isinstance(headers, dict):
        for (header, value) in headers.items():
            output += '    %s: %s\n' % (header, value)
    else:
        for pair in headers:
            output += '    %s: %s\n' % (pair[0], pair[1])
    return output

class HttpClient(object):
    """Performs HTTP requests using httplib."""
    debug = None

    def request(self, http_request):
        if False:
            for i in range(10):
                print('nop')
        return self._http_request(http_request.method, http_request.uri, http_request.headers, http_request._body_parts)
    Request = request

    def _get_connection(self, uri, headers=None):
        if False:
            for i in range(10):
                print('nop')
        'Opens a socket connection to the server to set up an HTTP request.\n\n        Args:\n          uri: The full URL for the request as a Uri object.\n          headers: A dict of string pairs containing the HTTP headers for the\n              request.\n        '
        connection = None
        if uri.scheme == 'https':
            if not uri.port:
                connection = http.client.HTTPSConnection(uri.host)
            else:
                connection = http.client.HTTPSConnection(uri.host, int(uri.port))
        elif not uri.port:
            connection = http.client.HTTPConnection(uri.host)
        else:
            connection = http.client.HTTPConnection(uri.host, int(uri.port))
        return connection

    def _http_request(self, method, uri, headers=None, body_parts=None):
        if False:
            while True:
                i = 10
        "Makes an HTTP request using httplib.\n\n        Args:\n          method: str example: 'GET', 'POST', 'PUT', 'DELETE', etc.\n          uri: str or atom.http_core.Uri\n          headers: dict of strings mapping to strings which will be sent as HTTP\n                   headers in the request.\n          body_parts: list of strings, objects with a read method, or objects\n                      which can be converted to strings using str. Each of these\n                      will be sent in order as the body of the HTTP request.\n        "
        if isinstance(uri, str):
            uri = Uri.parse_uri(uri)
        connection = self._get_connection(uri, headers=headers)
        if self.debug:
            connection.debuglevel = 1
        if connection.host != uri.host:
            connection.putrequest(method, str(uri))
        else:
            connection.putrequest(method, uri._get_relative_path())
        if uri.scheme == 'https' and int(uri.port or 443) == 443 and hasattr(connection, '_buffer') and isinstance(connection._buffer, list):
            header_line = 'Host: %s:443' % uri.host
            replacement_header_line = 'Host: %s' % uri.host
            try:
                connection._buffer[connection._buffer.index(header_line)] = replacement_header_line
            except ValueError:
                pass
        for (header_name, value) in headers.items():
            connection.putheader(header_name, value)
        connection.endheaders()
        if body_parts and [x for x in body_parts if x != '']:
            for part in body_parts:
                _send_data_part(part, connection)
        return connection.getresponse()

def _send_data_part(data, connection):
    if False:
        print('Hello World!')
    if isinstance(data, str):
        connection.send(data.encode())
    elif hasattr(data, 'read'):
        while 1:
            binarydata = data.read(100000)
            if binarydata == '':
                break
            connection.send(binarydata)
    else:
        connection.send(data)

class ProxiedHttpClient(HttpClient):

    def _get_connection(self, uri, headers=None):
        if False:
            for i in range(10):
                print('nop')
        proxy = None
        if uri.scheme == 'https':
            proxy = os.environ.get('https_proxy')
        elif uri.scheme == 'http':
            proxy = os.environ.get('http_proxy')
        if not proxy:
            return HttpClient._get_connection(self, uri, headers=headers)
        proxy_auth = _get_proxy_auth()
        if uri.scheme == 'https':
            import socket
            if proxy_auth:
                proxy_auth = 'Proxy-Authorization: %s' % proxy_auth
            port = uri.port
            if not port:
                port = 443
            proxy_connect = 'CONNECT %s:%s HTTP/1.0\r\n' % (uri.host, port)
            user_agent = ''
            if headers and 'User-Agent' in headers:
                user_agent = 'User-Agent: %s\r\n' % headers['User-Agent']
            proxy_pieces = '%s%s%s\r\n' % (proxy_connect, proxy_auth, user_agent)
            proxy_uri = Uri.parse_uri(proxy)
            if not proxy_uri.port:
                proxy_uri.port = '80'
            p_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            p_sock.connect((proxy_uri.host, int(proxy_uri.port)))
            p_sock.sendall(proxy_pieces.encode('utf-8'))
            response = ''
            while response.find('\r\n\r\n') == -1:
                response += p_sock.recv(8192).decode('utf-8')
            p_status = response.split()[1]
            if p_status != str(200):
                raise ProxyError('Error status=%s' % str(p_status))
            sslobj = None
            if ssl is not None:
                sslobj = ssl.wrap_socket(p_sock, None, None)
            else:
                sock_ssl = socket.ssl(p_sock, None, Nonesock_)
                sslobj = http.client.FakeSocket(p_sock, sock_ssl)
            connection = http.client.HTTPConnection(proxy_uri.host)
            connection.sock = sslobj
            return connection
        elif uri.scheme == 'http':
            proxy_uri = Uri.parse_uri(proxy)
            if not proxy_uri.port:
                proxy_uri.port = '80'
            if proxy_auth:
                headers['Proxy-Authorization'] = proxy_auth.strip()
            return http.client.HTTPConnection(proxy_uri.host, int(proxy_uri.port))
        return None

def _get_proxy_auth():
    if False:
        while True:
            i = 10
    import base64
    proxy_username = os.environ.get('proxy-username')
    if not proxy_username:
        proxy_username = os.environ.get('proxy_username')
    proxy_password = os.environ.get('proxy-password')
    if not proxy_password:
        proxy_password = os.environ.get('proxy_password')
    if proxy_username:
        user_auth = base64.b64encode(('%s:%s' % (proxy_username, proxy_password)).encode('utf-8'))
        return 'Basic %s\r\n' % user_auth.strip().decode('utf-8')
    else:
        return ''