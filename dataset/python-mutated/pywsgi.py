"""
A pure-Python, gevent-friendly WSGI server implementing HTTP/1.1.

The server is provided in :class:`WSGIServer`, but most of the actual
WSGI work is handled by :class:`WSGIHandler` --- a new instance is
created for each request. The server can be customized to use
different subclasses of :class:`WSGIHandler`.

.. important::

   This server is intended primarily for development and testing, and
   secondarily for other "safe" scenarios where it will not be exposed to
   potentially malicious input. The code has not been security audited,
   and is not intended for direct exposure to the public Internet. For production
   usage on the Internet, either choose a production-strength server such as
   gunicorn, or put a reverse proxy between gevent and the Internet.

.. versionchanged:: 23.9.0

   Complies more closely with the HTTP specification for chunked transfer encoding.
   In particular, we are much stricter about trailers, and trailers that
   are invalid (too long or featuring disallowed characters) forcibly close
   the connection to the client *after* the results have been sent.

   Trailers otherwise continue to be ignored and are not available to the
   WSGI application.

"""
from __future__ import absolute_import
import errno
from io import BytesIO
import string
import sys
import time
import traceback
from datetime import datetime
from urllib.parse import unquote
from gevent import socket
import gevent
from gevent.server import StreamServer
from gevent.hub import GreenletExit
from gevent._compat import reraise
from functools import partial
unquote_latin1 = partial(unquote, encoding='latin-1')
_no_undoc_members = True
__all__ = ['WSGIServer', 'WSGIHandler', 'LoggingLogAdapter', 'Environ', 'SecureEnviron', 'WSGISecureEnviron']
MAX_REQUEST_LINE = 8192
_WEEKDAYNAME = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
_MONTHNAME = (None, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
_HEX = string.hexdigits.encode('ascii')
_ALLOWED_TOKEN_CHARS = frozenset((c.encode('ascii') for c in "!#$%&'*+-.^_`|~0123456789")) | {c.encode('ascii') for c in string.ascii_letters}
assert b'A' in _ALLOWED_TOKEN_CHARS
_ERRORS = {}
_INTERNAL_ERROR_STATUS = '500 Internal Server Error'
_INTERNAL_ERROR_BODY = b'Internal Server Error'
_INTERNAL_ERROR_HEADERS = (('Content-Type', 'text/plain'), ('Connection', 'close'), ('Content-Length', str(len(_INTERNAL_ERROR_BODY))))
_ERRORS[500] = (_INTERNAL_ERROR_STATUS, _INTERNAL_ERROR_HEADERS, _INTERNAL_ERROR_BODY)
_BAD_REQUEST_STATUS = '400 Bad Request'
_BAD_REQUEST_BODY = ''
_BAD_REQUEST_HEADERS = (('Content-Type', 'text/plain'), ('Connection', 'close'), ('Content-Length', str(len(_BAD_REQUEST_BODY))))
_ERRORS[400] = (_BAD_REQUEST_STATUS, _BAD_REQUEST_HEADERS, _BAD_REQUEST_BODY)
_REQUEST_TOO_LONG_RESPONSE = b'HTTP/1.1 414 Request URI Too Long\r\nConnection: close\r\nContent-length: 0\r\n\r\n'
_BAD_REQUEST_RESPONSE = b'HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-length: 0\r\n\r\n'
_CONTINUE_RESPONSE = b'HTTP/1.1 100 Continue\r\n\r\n'

def format_date_time(timestamp):
    if False:
        for i in range(10):
            print('nop')
    (year, month, day, hh, mm, ss, wd, _y, _z) = time.gmtime(timestamp)
    value = '%s, %02d %3s %4d %02d:%02d:%02d GMT' % (_WEEKDAYNAME[wd], day, _MONTHNAME[month], year, hh, mm, ss)
    value = value.encode('latin-1')
    return value

class _InvalidClientInput(IOError):
    pass

class _InvalidClientRequest(ValueError):

    def __init__(self, message):
        if False:
            return 10
        ValueError.__init__(self, message)
        self.formatted_message = message

class Input(object):
    __slots__ = ('rfile', 'content_length', 'socket', 'position', 'chunked_input', 'chunk_length', '_chunked_input_error')

    def __init__(self, rfile, content_length, socket=None, chunked_input=False):
        if False:
            print('Hello World!')
        self.rfile = rfile
        self.content_length = content_length
        self.socket = socket
        self.position = 0
        self.chunked_input = chunked_input
        self.chunk_length = -1
        self._chunked_input_error = False

    def _discard(self):
        if False:
            i = 10
            return i + 15
        if self._chunked_input_error:
            return
        if self.socket is None and (self.position < (self.content_length or 0) or self.chunked_input):
            while 1:
                d = self.read(16384)
                if not d:
                    break

    def _send_100_continue(self):
        if False:
            i = 10
            return i + 15
        if self.socket is not None:
            self.socket.sendall(_CONTINUE_RESPONSE)
            self.socket = None

    def _do_read(self, length=None, use_readline=False):
        if False:
            for i in range(10):
                print('nop')
        if use_readline:
            reader = self.rfile.readline
        else:
            reader = self.rfile.read
        content_length = self.content_length
        if content_length is None:
            return b''
        self._send_100_continue()
        left = content_length - self.position
        if length is None:
            length = left
        elif length > left:
            length = left
        if not length:
            return b''
        try:
            read = reader(length)
        except OverflowError:
            if not use_readline:
                raise
            read = b''
            while len(read) < length and (not read.endswith(b'\n')):
                read += reader(MAX_REQUEST_LINE)
        self.position += len(read)
        if len(read) < length:
            if use_readline and (not read.endswith(b'\n')) or not use_readline:
                raise IOError('unexpected end of file while reading request at position %s' % (self.position,))
        return read

    def __read_chunk_length(self, rfile):
        if False:
            for i in range(10):
                print('nop')
        buf = BytesIO()
        while 1:
            char = rfile.read(1)
            if not char:
                self._chunked_input_error = True
                raise _InvalidClientInput('EOF before chunk end reached')
            if char in (b'\r', b';'):
                break
            if char not in _HEX:
                self._chunked_input_error = True
                raise _InvalidClientInput('Non-hex data', char)
            buf.write(char)
            if buf.tell() > 16:
                self._chunked_input_error = True
                raise _InvalidClientInput('Chunk-size too large.')
        if char == b';':
            i = 0
            while i < MAX_REQUEST_LINE:
                char = rfile.read(1)
                if char == b'\r':
                    break
                i += 1
            else:
                self._chunked_input_error = True
                raise _InvalidClientInput('Too large chunk extension')
        if char == b'\r':
            self.__read_chunk_size_crlf(rfile, newline_only=True)
            result = int(buf.getvalue(), 16)
            if result == 0:
                while self.__read_chunk_trailer(rfile):
                    pass
            return result

    def __read_chunk_trailer(self, rfile):
        if False:
            print('Hello World!')
        i = 0
        empty = True
        seen_field_name = False
        while i < MAX_REQUEST_LINE:
            char = rfile.read(1)
            if char == b'\r':
                self.__read_chunk_size_crlf(rfile, newline_only=True)
                break
            empty = False
            if char == b':' and i > 0:
                seen_field_name = True
            if not seen_field_name and char not in _ALLOWED_TOKEN_CHARS:
                raise _InvalidClientInput('Invalid token character: %r' % (char,))
            i += 1
        else:
            self._chunked_input_error = True
            raise _InvalidClientInput('Too large chunk trailer')
        return not empty

    def __read_chunk_size_crlf(self, rfile, newline_only=False):
        if False:
            for i in range(10):
                print('nop')
        if not newline_only:
            char = rfile.read(1)
            if char != b'\r':
                self._chunked_input_error = True
                raise _InvalidClientInput("Line didn't end in CRLF: %r" % (char,))
        char = rfile.read(1)
        if char != b'\n':
            self._chunked_input_error = True
            raise _InvalidClientInput("Line didn't end in LF: %r" % (char,))

    def _chunked_read(self, length=None, use_readline=False):
        if False:
            return 10
        rfile = self.rfile
        self._send_100_continue()
        if length == 0:
            return b''
        if use_readline:
            reader = self.rfile.readline
        else:
            reader = self.rfile.read
        response = []
        while self.chunk_length != 0:
            maxreadlen = self.chunk_length - self.position
            if length is not None and length < maxreadlen:
                maxreadlen = length
            if maxreadlen > 0:
                data = reader(maxreadlen)
                if not data:
                    self.chunk_length = 0
                    self._chunked_input_error = True
                    raise IOError('unexpected end of file while parsing chunked data')
                datalen = len(data)
                response.append(data)
                self.position += datalen
                if self.chunk_length == self.position:
                    self.__read_chunk_size_crlf(rfile)
                if length is not None:
                    length -= datalen
                    if length == 0:
                        break
                if use_readline and data[-1] == b'\n'[0]:
                    break
            else:
                self.chunk_length = self.__read_chunk_length(rfile)
                self.position = 0
        return b''.join(response)

    def read(self, length=None):
        if False:
            while True:
                i = 10
        if length is not None and length < 0:
            length = None
        if self.chunked_input:
            return self._chunked_read(length)
        return self._do_read(length)

    def readline(self, size=None):
        if False:
            while True:
                i = 10
        if size is not None and size < 0:
            size = None
        if self.chunked_input:
            return self._chunked_read(size, True)
        return self._do_read(size, use_readline=True)

    def readlines(self, hint=None):
        if False:
            return 10
        return list(self)

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    def next(self):
        if False:
            return 10
        line = self.readline()
        if not line:
            raise StopIteration
        return line
    __next__ = next
try:
    import mimetools
    headers_factory = mimetools.Message
except ImportError:
    from http import client

    class OldMessage(client.HTTPMessage):

        def __init__(self, **kwargs):
            if False:
                return 10
            super(client.HTTPMessage, self).__init__(**kwargs)
            self.status = ''

        def getheader(self, name, default=None):
            if False:
                for i in range(10):
                    print('nop')
            return self.get(name, default)

        @property
        def headers(self):
            if False:
                i = 10
                return i + 15
            for (key, value) in self._headers:
                yield ('%s: %s\r\n' % (key, value))

        @property
        def typeheader(self):
            if False:
                return 10
            return self.get('content-type')

    def headers_factory(fp, *args):
        if False:
            print('Hello World!')
        try:
            ret = client.parse_headers(fp, _class=OldMessage)
        except client.LineTooLong:
            ret = OldMessage()
            ret.status = 'Line too long'
        return ret

class WSGIHandler(object):
    """
    Handles HTTP requests from a socket, creates the WSGI environment, and
    interacts with the WSGI application.

    This is the default value of :attr:`WSGIServer.handler_class`.
    This class may be subclassed carefully, and that class set on a
    :class:`WSGIServer` instance through a keyword argument at
    construction time.

    Instances are constructed with the same arguments as passed to the
    server's :meth:`WSGIServer.handle` method followed by the server
    itself. The application and environment are obtained from the server.

    """
    protocol_version = 'HTTP/1.1'

    def MessageClass(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return headers_factory(*args)
    status = None
    _orig_status = None
    response_headers = None
    code = None
    provided_date = None
    provided_content_length = None
    close_connection = False
    time_start = 0
    time_finish = 0
    headers_sent = False
    response_use_chunked = False
    connection_upgraded = False
    environ = None
    application = None
    requestline = None
    response_length = 0
    result = None
    wsgi_input = None
    content_length = 0
    headers = headers_factory(BytesIO())
    request_version = None
    command = None
    path = None

    def __init__(self, sock, address, server, rfile=None):
        if False:
            return 10
        self.socket = sock
        self.client_address = address
        self.server = server
        if rfile is None:
            self.rfile = sock.makefile('rb', -1)
        else:
            self.rfile = rfile

    def handle(self):
        if False:
            i = 10
            return i + 15
        '\n        The main request handling method, called by the server.\n\n        This method runs a request handling loop, calling\n        :meth:`handle_one_request` until all requests on the\n        connection have been handled (that is, it implements\n        keep-alive).\n        '
        try:
            while self.socket is not None:
                self.time_start = time.time()
                self.time_finish = 0
                result = self.handle_one_request()
                if result is None:
                    break
                if result is True:
                    continue
                (self.status, response_body) = result
                self.socket.sendall(response_body)
                if self.time_finish == 0:
                    self.time_finish = time.time()
                self.log_request()
                break
        finally:
            if self.socket is not None:
                _sock = getattr(self.socket, '_sock', None)
                try:
                    if _sock:
                        try:
                            _sock.recv(16384)
                        finally:
                            _sock.close()
                    self.socket.close()
                except socket.error:
                    pass
            self.__dict__.pop('socket', None)
            self.__dict__.pop('rfile', None)
            self.__dict__.pop('wsgi_input', None)

    def _check_http_version(self):
        if False:
            i = 10
            return i + 15
        version_str = self.request_version
        if not version_str.startswith('HTTP/'):
            return False
        version = tuple((int(x) for x in version_str[5:].split('.')))
        if version[1] < 0 or version < (0, 9) or version >= (2, 0):
            return False
        return True

    def read_request(self, raw_requestline):
        if False:
            for i in range(10):
                print('nop')
        "\n        Parse the incoming request.\n\n        Parses various headers into ``self.headers`` using\n        :attr:`MessageClass`. Other attributes that are set upon a successful\n        return of this method include ``self.content_length`` and ``self.close_connection``.\n\n        :param str raw_requestline: A native :class:`str` representing\n           the request line. A processed version of this will be stored\n           into ``self.requestline``.\n\n        :raises ValueError: If the request is invalid. This error will\n           not be logged as a traceback (because it's a client issue, not a server problem).\n        :return: A boolean value indicating whether the request was successfully parsed.\n           This method should either return a true value or have raised a ValueError\n           with details about the parsing error.\n\n        .. versionchanged:: 1.1b6\n           Raise the previously documented :exc:`ValueError` in more cases instead of returning a\n           false value; this allows subclasses more opportunity to customize behaviour.\n        "
        self.requestline = raw_requestline.rstrip()
        words = self.requestline.split()
        if len(words) == 3:
            (self.command, self.path, self.request_version) = words
            if not self._check_http_version():
                raise _InvalidClientRequest('Invalid http version: %r' % (raw_requestline,))
        elif len(words) == 2:
            (self.command, self.path) = words
            if self.command != 'GET':
                raise _InvalidClientRequest('Expected GET method; Got command=%r; path=%r; raw=%r' % (self.command, self.path, raw_requestline))
            self.request_version = 'HTTP/0.9'
        else:
            raise _InvalidClientRequest('Invalid HTTP method: %r' % (raw_requestline,))
        self.headers = self.MessageClass(self.rfile, 0)
        if self.headers.status:
            raise _InvalidClientRequest('Invalid headers status: %r' % (self.headers.status,))
        if self.headers.get('transfer-encoding', '').lower() == 'chunked':
            try:
                del self.headers['content-length']
            except KeyError:
                pass
        content_length = self.headers.get('content-length')
        if content_length is not None:
            content_length = int(content_length)
            if content_length < 0:
                raise _InvalidClientRequest('Invalid Content-Length: %r' % (content_length,))
            if content_length and self.command in ('HEAD',):
                raise _InvalidClientRequest('Unexpected Content-Length')
        self.content_length = content_length
        if self.request_version == 'HTTP/1.1':
            conntype = self.headers.get('Connection', '').lower()
            self.close_connection = conntype == 'close'
        elif self.request_version == 'HTTP/1.0':
            conntype = self.headers.get('Connection', 'close').lower()
            self.close_connection = conntype != 'keep-alive'
        else:
            self.close_connection = True
        return True
    _print_unexpected_exc = staticmethod(traceback.print_exc)

    def log_error(self, msg, *args):
        if False:
            return 10
        if not args:
            message = msg
        else:
            try:
                message = msg % args
            except Exception:
                self._print_unexpected_exc()
                message = '%r %r' % (msg, args)
        try:
            message = '%s: %s' % (self.socket, message)
        except Exception:
            pass
        try:
            self.server.error_log.write(message + '\n')
        except Exception:
            self._print_unexpected_exc()

    def read_requestline(self):
        if False:
            print('Hello World!')
        '\n        Read and return the HTTP request line.\n\n        Under both Python 2 and 3, this should return the native\n        ``str`` type; under Python 3, this probably means the bytes read\n        from the network need to be decoded (using the ISO-8859-1 charset, aka\n        latin-1).\n        '
        line = self.rfile.readline(MAX_REQUEST_LINE)
        line = line.decode('latin-1')
        return line

    def handle_one_request(self):
        if False:
            return 10
        '\n        Handles one HTTP request using ``self.socket`` and ``self.rfile``.\n\n        Each invocation of this method will do several things, including (but not limited to):\n\n        - Read the request line using :meth:`read_requestline`;\n        - Read the rest of the request, including headers, with :meth:`read_request`;\n        - Construct a new WSGI environment in ``self.environ`` using :meth:`get_environ`;\n        - Store the application in ``self.application``, retrieving it from the server;\n        - Handle the remainder of the request, including invoking the application,\n          with :meth:`handle_one_response`\n\n        There are several possible return values to indicate the state\n        of the client connection:\n\n        - ``None``\n            The client connection is already closed or should\n            be closed because the WSGI application or client set the\n            ``Connection: close`` header. The request handling\n            loop should terminate and perform cleanup steps.\n        - (status, body)\n            An HTTP status and body tuple. The request was in error,\n            as detailed by the status and body. The request handling\n            loop should terminate, close the connection, and perform\n            cleanup steps. Note that the ``body`` is the complete contents\n            to send to the client, including all headers and the initial\n            status line.\n        - ``True``\n            The literal ``True`` value. The request was successfully handled\n            and the response sent to the client by :meth:`handle_one_response`.\n            The connection remains open to process more requests and the connection\n            handling loop should call this method again. This is the typical return\n            value.\n\n        .. seealso:: :meth:`handle`\n\n        .. versionchanged:: 1.1b6\n           Funnel exceptions having to do with invalid HTTP requests through\n           :meth:`_handle_client_error` to allow subclasses to customize. Note that\n           this is experimental and may change in the future.\n        '
        if self.rfile.closed:
            return
        try:
            self.requestline = self.read_requestline()
            if isinstance(self.requestline, bytes):
                self.requestline = self.requestline.decode('latin-1')
        except socket.error:
            return
        if not self.requestline:
            return
        self.response_length = 0
        if len(self.requestline) >= MAX_REQUEST_LINE:
            return ('414', _REQUEST_TOO_LONG_RESPONSE)
        try:
            if not self.read_request(self.requestline):
                return ('400', _BAD_REQUEST_RESPONSE)
        except Exception as ex:
            return self._handle_client_error(ex)
        self.environ = self.get_environ()
        self.application = self.server.application
        self.handle_one_response()
        if self.close_connection:
            return
        if self.rfile.closed:
            return
        return True

    def _connection_upgrade_requested(self):
        if False:
            i = 10
            return i + 15
        if self.headers.get('Connection', '').lower() == 'upgrade':
            return True
        if self.headers.get('Upgrade', '').lower() == 'websocket':
            return True
        return False

    def finalize_headers(self):
        if False:
            i = 10
            return i + 15
        if self.provided_date is None:
            self.response_headers.append((b'Date', format_date_time(time.time())))
        self.connection_upgraded = self.code == 101
        if self.code not in (304, 204):
            if self.provided_content_length is None:
                if hasattr(self.result, '__len__'):
                    total_len = sum((len(chunk) for chunk in self.result))
                    total_len_str = str(total_len)
                    total_len_str = total_len_str.encode('latin-1')
                    self.response_headers.append((b'Content-Length', total_len_str))
                else:
                    self.response_use_chunked = not self.connection_upgraded and self.request_version != 'HTTP/1.0'
                    if self.response_use_chunked:
                        self.response_headers.append((b'Transfer-Encoding', b'chunked'))

    def _sendall(self, data):
        if False:
            return 10
        try:
            self.socket.sendall(data)
        except socket.error as ex:
            self.status = 'socket error: %s' % ex
            if self.code > 0:
                self.code = -self.code
            raise
        self.response_length += len(data)

    def _write(self, data, _bytearray=bytearray):
        if False:
            return 10
        if not data:
            return
        if self.response_use_chunked:
            header_str = b'%x\r\n' % len(data)
            towrite = _bytearray(header_str)
            towrite += data
            towrite += b'\r\n'
            self._sendall(towrite)
        else:
            self._sendall(data)
    ApplicationError = AssertionError

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self.code in (304, 204) and data:
            raise self.ApplicationError('The %s response must have no body' % self.code)
        if self.headers_sent:
            self._write(data)
        else:
            if not self.status:
                raise self.ApplicationError('The application did not call start_response()')
            self._write_with_headers(data)

    def _write_with_headers(self, data):
        if False:
            return 10
        self.headers_sent = True
        self.finalize_headers()
        towrite = bytearray(b'HTTP/1.1 ')
        towrite += self.status
        towrite += b'\r\n'
        for (header, value) in self.response_headers:
            towrite += header
            towrite += b': '
            towrite += value
            towrite += b'\r\n'
        towrite += b'\r\n'
        self._sendall(towrite)
        self._write(data)

    def start_response(self, status, headers, exc_info=None):
        if False:
            i = 10
            return i + 15
        '\n         .. versionchanged:: 1.2a1\n            Avoid HTTP header injection by raising a :exc:`ValueError`\n            if *status* or any *header* name or value contains a carriage\n            return or newline.\n         .. versionchanged:: 1.1b5\n            Pro-actively handle checking the encoding of the status line\n            and headers during this method. On Python 2, avoid some\n            extra encodings.\n        '
        if exc_info:
            try:
                if self.headers_sent:
                    reraise(*exc_info)
            finally:
                exc_info = None
        response_headers = []
        header = None
        value = None
        try:
            for (header, value) in headers:
                if not isinstance(header, str):
                    raise UnicodeError('The header must be a native string', header, value)
                if not isinstance(value, str):
                    raise UnicodeError('The value must be a native string', header, value)
                if '\r' in header or '\n' in header:
                    raise ValueError('carriage return or newline in header name', header)
                if '\r' in value or '\n' in value:
                    raise ValueError('carriage return or newline in header value', value)
                response_headers.append((header.encode('latin-1'), value.encode('latin-1')))
        except UnicodeEncodeError:
            raise UnicodeError('Non-latin1 header', repr(header), repr(value))
        if not isinstance(status, str):
            raise UnicodeError('The status string must be a native string')
        if '\r' in status or '\n' in status:
            raise ValueError('carriage return or newline in status', status)
        code = int(status.split(' ', 1)[0])
        self.status = status.encode('latin-1')
        self._orig_status = status
        self.response_headers = response_headers
        self.code = code
        provided_connection = None
        self.provided_date = None
        self.provided_content_length = None
        for (header, value) in headers:
            header = header.lower()
            if header == 'connection':
                provided_connection = value
            elif header == 'date':
                self.provided_date = value
            elif header == 'content-length':
                self.provided_content_length = value
        if self.request_version == 'HTTP/1.0' and provided_connection is None:
            conntype = b'close' if self.close_connection else b'keep-alive'
            response_headers.append((b'Connection', conntype))
        elif provided_connection == 'close':
            self.close_connection = True
        if self.code in (304, 204):
            if self.provided_content_length is not None and self.provided_content_length != '0':
                msg = 'Invalid Content-Length for %s response: %r (must be absent or zero)' % (self.code, self.provided_content_length)
                msg = msg.encode('latin-1')
                raise self.ApplicationError(msg)
        return self.write

    def log_request(self):
        if False:
            i = 10
            return i + 15
        self.server.log.write(self.format_request() + '\n')

    def format_request(self):
        if False:
            return 10
        now = datetime.now().replace(microsecond=0)
        length = self.response_length or '-'
        if self.time_finish:
            delta = '%.6f' % (self.time_finish - self.time_start)
        else:
            delta = '-'
        client_address = self.client_address[0] if isinstance(self.client_address, tuple) else self.client_address
        return '%s - - [%s] "%s" %s %s %s' % (client_address or '-', now, self.requestline or '', (self._orig_status or self.status or '000').split()[0], length, delta)

    def process_result(self):
        if False:
            while True:
                i = 10
        for data in self.result:
            if data:
                self.write(data)
        if self.status and (not self.headers_sent):
            self.write(b'')
        if self.response_use_chunked:
            self._sendall(b'0\r\n\r\n')

    def run_application(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.result is None
        try:
            self.result = self.application(self.environ, self.start_response)
            self.process_result()
        finally:
            close = getattr(self.result, 'close', None)
            try:
                if close is not None:
                    close()
            finally:
                close = None
                self.result = None
    ignored_socket_errors = (errno.EPIPE, errno.ECONNRESET)
    try:
        ignored_socket_errors += (errno.WSAECONNABORTED,)
    except AttributeError:
        pass

    def handle_one_response(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Invoke the application to produce one response.\n\n        This is called by :meth:`handle_one_request` after all the\n        state for the request has been established. It is responsible\n        for error handling.\n        '
        self.time_start = time.time()
        self.status = None
        self.headers_sent = False
        self.result = None
        self.response_use_chunked = False
        self.connection_upgraded = False
        self.response_length = 0
        try:
            try:
                self.run_application()
            finally:
                try:
                    self.wsgi_input._discard()
                except _InvalidClientInput:
                    raise
                except socket.error:
                    pass
        except _InvalidClientInput as ex:
            self._handle_client_error(ex)
            self.close_connection = True
            self._send_error_response_if_possible(400)
        except socket.error as ex:
            if ex.args[0] in self.ignored_socket_errors:
                self.close_connection = True
            else:
                self.handle_error(*sys.exc_info())
        except:
            self.handle_error(*sys.exc_info())
        finally:
            self.time_finish = time.time()
            self.log_request()

    def _send_error_response_if_possible(self, error_code):
        if False:
            return 10
        if self.response_length:
            self.close_connection = True
        else:
            (status, headers, body) = _ERRORS[error_code]
            try:
                self.start_response(status, headers[:])
                self.write(body)
            except socket.error:
                self.close_connection = True

    def _log_error(self, t, v, tb):
        if False:
            while True:
                i = 10
        if not issubclass(t, GreenletExit):
            context = self.environ
            if not isinstance(context, self.server.secure_environ_class):
                context = self.server.secure_environ_class(context)
            self.server.loop.handle_error(context, t, v, tb)

    def handle_error(self, t, v, tb):
        if False:
            return 10
        self._log_error(t, v, tb)
        t = v = tb = None
        self._send_error_response_if_possible(500)

    def _handle_client_error(self, ex):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(ex, (ValueError, _InvalidClientInput)):
            traceback.print_exc()
        if isinstance(ex, _InvalidClientRequest):
            self.log_error('(from %s) %s', self.client_address, ex.formatted_message)
        else:
            self.log_error('Invalid request (from %s): %s', self.client_address, str(ex) or ex.__class__.__name__)
        return ('400', _BAD_REQUEST_RESPONSE)

    def _headers(self):
        if False:
            print('Hello World!')
        key = None
        value = None
        IGNORED_KEYS = (None, 'CONTENT_TYPE', 'CONTENT_LENGTH')
        for header in self.headers.headers:
            if key is not None and header[:1] in ' \t':
                value += header
                continue
            if key not in IGNORED_KEYS:
                yield ('HTTP_' + key, value.strip())
            (key, value) = header.split(':', 1)
            if '_' in key:
                key = None
            else:
                key = key.replace('-', '_').upper()
        if key not in IGNORED_KEYS:
            yield ('HTTP_' + key, value.strip())

    def get_environ(self):
        if False:
            while True:
                i = 10
        '\n        Construct and return a new WSGI environment dictionary for a specific request.\n\n        This should begin with asking the server for the base environment\n        using :meth:`WSGIServer.get_environ`, and then proceed to add the\n        request specific values.\n\n        By the time this method is invoked the request line and request shall have\n        been parsed and ``self.headers`` shall be populated.\n        '
        env = self.server.get_environ()
        env['REQUEST_METHOD'] = self.command
        env['SCRIPT_NAME'] = ''
        (path, query) = self.path.split('?', 1) if '?' in self.path else (self.path, '')
        env['PATH_INFO'] = unquote_latin1(path)
        env['QUERY_STRING'] = query
        if self.headers.typeheader is not None:
            env['CONTENT_TYPE'] = self.headers.typeheader
        length = self.headers.getheader('content-length')
        if length:
            env['CONTENT_LENGTH'] = length
        env['SERVER_PROTOCOL'] = self.request_version
        client_address = self.client_address
        if isinstance(client_address, tuple):
            env['REMOTE_ADDR'] = str(client_address[0])
            env['REMOTE_PORT'] = str(client_address[1])
        for (key, value) in self._headers():
            if key in env:
                if 'COOKIE' in key:
                    env[key] += '; ' + value
                else:
                    env[key] += ',' + value
            else:
                env[key] = value
        sock = self.socket if env.get('HTTP_EXPECT') == '100-continue' else None
        chunked = env.get('HTTP_TRANSFER_ENCODING', '').lower() == 'chunked'
        handling_reads = not self._connection_upgrade_requested()
        self.wsgi_input = Input(self.rfile, self.content_length, socket=sock, chunked_input=chunked)
        env['wsgi.input'] = self.wsgi_input if handling_reads else self.rfile
        env['wsgi.input_terminated'] = handling_reads
        return env

class _NoopLog(object):

    def write(self, *args, **kwargs):
        if False:
            return 10
        return

    def flush(self):
        if False:
            while True:
                i = 10
        pass

    def writelines(self, *args, **kwargs):
        if False:
            return 10
        pass

class LoggingLogAdapter(object):
    """
    An adapter for :class:`logging.Logger` instances
    to let them be used with :class:`WSGIServer`.

    .. warning:: Unless the entire process is monkey-patched at a very
        early part of the lifecycle (before logging is configured),
        loggers are likely to not be gevent-cooperative. For example,
        the socket and syslog handlers use the socket module in a way
        that can block, and most handlers acquire threading locks.

    .. warning:: It *may* be possible for the logging functions to be
       called in the :class:`gevent.Hub` greenlet. Code running in the
       hub greenlet cannot use any gevent blocking functions without triggering
       a ``LoopExit``.

    .. versionadded:: 1.1a3

    .. versionchanged:: 1.1b6
       Attributes not present on this object are proxied to the underlying
       logger instance. This permits using custom :class:`~logging.Logger`
       subclasses (or indeed, even duck-typed objects).

    .. versionchanged:: 1.1
       Strip trailing newline characters on the message passed to :meth:`write`
       because log handlers will usually add one themselves.
    """
    __slots__ = ('_logger', '_level')

    def __init__(self, logger, level=20):
        if False:
            i = 10
            return i + 15
        '\n        Write information to the *logger* at the given *level* (default to INFO).\n        '
        self._logger = logger
        self._level = level

    def write(self, msg):
        if False:
            i = 10
            return i + 15
        if msg and msg.endswith('\n'):
            msg = msg[:-1]
        self._logger.log(self._level, msg)

    def flush(self):
        if False:
            i = 10
            return i + 15
        'No-op; required to be a file-like object'

    def writelines(self, lines):
        if False:
            print('Hello World!')
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._logger, name)

    def __setattr__(self, name, value):
        if False:
            print('Hello World!')
        if name not in LoggingLogAdapter.__slots__:
            setattr(self._logger, name, value)
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if False:
            return 10
        delattr(self._logger, name)

class Environ(dict):
    """
    A base class that can be used for WSGI environment objects.

    Provisional API.

    .. versionadded:: 1.2a1
    """
    __slots__ = ()

    def copy(self):
        if False:
            return 10
        return self.__class__(self)
    if not hasattr(dict, 'iteritems'):

        def iteritems(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.items()

    def __reduce_ex__(self, proto):
        if False:
            i = 10
            return i + 15
        return (dict, (), None, None, iter(self.iteritems()))

class SecureEnviron(Environ):
    """
    An environment that does not print its keys and values
    by default.

    Provisional API.

    This is intended to keep potentially sensitive information like
    HTTP authorization and cookies from being inadvertently printed
    or logged.

    For debugging, each instance can have its *secure_repr* attribute
    set to ``False``, which will cause it to print like a normal dict.

    When *secure_repr* is ``True`` (the default), then the value of
    the *whitelist_keys* attribute is consulted; if this value is
    true-ish, it should be a container (something that responds to
    ``in``) of key names (typically a list or set). Keys and values in
    this dictionary that are in *whitelist_keys* will then be printed,
    while all other values will be masked. These values may be
    customized on the class by setting the *default_secure_repr* and
    *default_whitelist_keys*, respectively::

        >>> environ = SecureEnviron(key='value')
        >>> environ # doctest: +ELLIPSIS
        <pywsgi.SecureEnviron dict (keys: 1) at ...

    If we whitelist the key, it gets printed::

        >>> environ.whitelist_keys = {'key'}
        >>> environ
        {'key': 'value'}

    A non-whitelisted key (*only*, to avoid doctest issues) is masked::

        >>> environ['secure'] = 'secret'; del environ['key']
        >>> environ
        {'secure': '<MASKED>'}

    We can turn it off entirely for the instance::

        >>> environ.secure_repr = False
        >>> environ
        {'secure': 'secret'}

    We can also customize it at the class level (here we use a new
    class to be explicit and to avoid polluting the true default
    values; we would set this class to be the ``environ_class`` of the
    server)::

        >>> class MyEnviron(SecureEnviron):
        ...    default_whitelist_keys = ('key',)
        ...
        >>> environ = MyEnviron({'key': 'value'})
        >>> environ
        {'key': 'value'}

    .. versionadded:: 1.2a1
    """
    default_secure_repr = True
    default_whitelist_keys = ()
    default_print_masked_keys = True
    __slots__ = ('secure_repr', 'whitelist_keys', 'print_masked_keys')

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        if name in SecureEnviron.__slots__:
            return getattr(type(self), 'default_' + name)
        raise AttributeError(name)

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self.secure_repr:
            whitelist = self.whitelist_keys
            print_masked = self.print_masked_keys
            if whitelist:
                safe = {k: self[k] if k in whitelist else '<MASKED>' for k in self if k in whitelist or print_masked}
                safe_repr = repr(safe)
                if not print_masked and len(safe) != len(self):
                    safe_repr = safe_repr[:-1] + ', (hidden keys: %d)}' % (len(self) - len(safe))
                return safe_repr
            return '<pywsgi.SecureEnviron dict (keys: %d) at %s>' % (len(self), id(self))
        return Environ.__repr__(self)
    __str__ = __repr__

class WSGISecureEnviron(SecureEnviron):
    """
    Specializes the default list of whitelisted keys to a few
    common WSGI variables.

    Example::

       >>> environ = WSGISecureEnviron(REMOTE_ADDR='::1', HTTP_AUTHORIZATION='secret')
       >>> environ
       {'REMOTE_ADDR': '::1', (hidden keys: 1)}
       >>> import pprint
       >>> pprint.pprint(environ)
       {'REMOTE_ADDR': '::1', (hidden keys: 1)}
       >>> print(pprint.pformat(environ))
       {'REMOTE_ADDR': '::1', (hidden keys: 1)}
    """
    default_whitelist_keys = ('REMOTE_ADDR', 'REMOTE_PORT', 'HTTP_HOST')
    default_print_masked_keys = False

class WSGIServer(StreamServer):
    """
    A WSGI server based on :class:`StreamServer` that supports HTTPS.


    :keyword log: If given, an object with a ``write`` method to which
        request (access) logs will be written. If not given, defaults
        to :obj:`sys.stderr`. You may pass ``None`` to disable request
        logging. You may use a wrapper, around e.g., :mod:`logging`,
        to support objects that don't implement a ``write`` method.
        (If you pass a :class:`~logging.Logger` instance, or in
        general something that provides a ``log`` method but not a
        ``write`` method, such a wrapper will automatically be created
        and it will be logged to at the :data:`~logging.INFO` level.)

    :keyword error_log: If given, a file-like object with ``write``,
        ``writelines`` and ``flush`` methods to which error logs will
        be written. If not given, defaults to :obj:`sys.stderr`. You
        may pass ``None`` to disable error logging (not recommended).
        You may use a wrapper, around e.g., :mod:`logging`, to support
        objects that don't implement the proper methods. This
        parameter will become the value for ``wsgi.errors`` in the
        WSGI environment (if not already set). (As with *log*,
        wrappers for :class:`~logging.Logger` instances and the like
        will be created automatically and logged to at the :data:`~logging.ERROR`
        level.)

    .. seealso::

        :class:`LoggingLogAdapter`
            See important warnings before attempting to use :mod:`logging`.

    .. versionchanged:: 1.1a3
        Added the ``error_log`` parameter, and set ``wsgi.errors`` in the WSGI
        environment to this value.
    .. versionchanged:: 1.1a3
        Add support for passing :class:`logging.Logger` objects to the ``log`` and
        ``error_log`` arguments.
    .. versionchanged:: 20.6.0
        Passing a ``handle`` kwarg to the constructor is now officially deprecated.
    """
    handler_class = WSGIHandler
    log = None
    error_log = None
    environ_class = dict
    secure_environ_class = WSGISecureEnviron
    base_env = {'GATEWAY_INTERFACE': 'CGI/1.1', 'SERVER_SOFTWARE': 'gevent/%d.%d Python/%d.%d' % (gevent.version_info[:2] + sys.version_info[:2]), 'SCRIPT_NAME': '', 'wsgi.version': (1, 0), 'wsgi.multithread': False, 'wsgi.multiprocess': False, 'wsgi.run_once': False}

    def __init__(self, listener, application=None, backlog=None, spawn='default', log='default', error_log='default', handler_class=None, environ=None, **ssl_args):
        if False:
            for i in range(10):
                print('nop')
        if 'handle' in ssl_args:
            import warnings
            warnings.warn("Passing 'handle' kwarg to WSGIServer is deprecated. Did you mean application?", DeprecationWarning, stacklevel=2)
        StreamServer.__init__(self, listener, backlog=backlog, spawn=spawn, **ssl_args)
        if application is not None:
            self.application = application
        if handler_class is not None:
            self.handler_class = handler_class

        def _make_log(l, level=20):
            if False:
                i = 10
                return i + 15
            if l == 'default':
                return sys.stderr
            if l is None:
                return _NoopLog()
            if not hasattr(l, 'write') and hasattr(l, 'log'):
                return LoggingLogAdapter(l, level)
            return l
        self.log = _make_log(log)
        self.error_log = _make_log(error_log, 40)
        self.set_environ(environ)
        self.set_max_accept()

    def set_environ(self, environ=None):
        if False:
            return 10
        if environ is not None:
            self.environ = environ
        environ_update = getattr(self, 'environ', None)
        self.environ = self.environ_class(self.base_env)
        if self.ssl_enabled:
            self.environ['wsgi.url_scheme'] = 'https'
        else:
            self.environ['wsgi.url_scheme'] = 'http'
        if environ_update is not None:
            self.environ.update(environ_update)
        if self.environ.get('wsgi.errors') is None:
            self.environ['wsgi.errors'] = self.error_log

    def set_max_accept(self):
        if False:
            print('Hello World!')
        if self.environ.get('wsgi.multiprocess'):
            self.max_accept = 1

    def get_environ(self):
        if False:
            i = 10
            return i + 15
        return self.environ_class(self.environ)

    def init_socket(self):
        if False:
            i = 10
            return i + 15
        StreamServer.init_socket(self)
        self.update_environ()

    def update_environ(self):
        if False:
            return 10
        '\n        Called before the first request is handled to fill in WSGI environment values.\n\n        This includes getting the correct server name and port.\n        '
        address = self.address
        if isinstance(address, tuple):
            if 'SERVER_NAME' not in self.environ:
                try:
                    name = socket.getfqdn(address[0])
                except socket.error:
                    name = str(address[0])
                if not isinstance(name, str):
                    name = name.decode('ascii')
                self.environ['SERVER_NAME'] = name
            self.environ.setdefault('SERVER_PORT', str(address[1]))
        else:
            self.environ.setdefault('SERVER_NAME', '')
            self.environ.setdefault('SERVER_PORT', '')

    def handle(self, sock, address):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create an instance of :attr:`handler_class` to handle the request.\n\n        This method blocks until the handler returns.\n        '
        handler = self.handler_class(sock, address, self)
        handler.handle()

def _main():
    if False:
        for i in range(10):
            print('nop')
    from gevent import monkey
    monkey.patch_all()
    import argparse
    import importlib
    parser = argparse.ArgumentParser()
    parser.add_argument('app', help='dotted name of WSGI app callable [module:callable]')
    parser.add_argument('-b', '--bind', help='The socket to bind', default=':8080')
    args = parser.parse_args()
    (module_name, app_name) = args.app.split(':')
    module = importlib.import_module(module_name)
    app = getattr(module, app_name)
    bind = args.bind
    server = WSGIServer(bind, app)
    server.serve_forever()
if __name__ == '__main__':
    _main()