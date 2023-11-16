__author__ = 'Allan Saddi <allan@saddi.com>'
__version__ = '$Revision$'
import sys
import select
import struct
import socket
import errno
import types
__all__ = ['FCGIApp']
FCGI_LISTENSOCK_FILENO = 0
FCGI_HEADER_LEN = 8
FCGI_VERSION_1 = 1
FCGI_BEGIN_REQUEST = 1
FCGI_ABORT_REQUEST = 2
FCGI_END_REQUEST = 3
FCGI_PARAMS = 4
FCGI_STDIN = 5
FCGI_STDOUT = 6
FCGI_STDERR = 7
FCGI_DATA = 8
FCGI_GET_VALUES = 9
FCGI_GET_VALUES_RESULT = 10
FCGI_UNKNOWN_TYPE = 11
FCGI_MAXTYPE = FCGI_UNKNOWN_TYPE
FCGI_NULL_REQUEST_ID = 0
FCGI_KEEP_CONN = 1
FCGI_RESPONDER = 1
FCGI_AUTHORIZER = 2
FCGI_FILTER = 3
FCGI_REQUEST_COMPLETE = 0
FCGI_CANT_MPX_CONN = 1
FCGI_OVERLOADED = 2
FCGI_UNKNOWN_ROLE = 3
FCGI_MAX_CONNS = 'FCGI_MAX_CONNS'
FCGI_MAX_REQS = 'FCGI_MAX_REQS'
FCGI_MPXS_CONNS = 'FCGI_MPXS_CONNS'
FCGI_Header = '!BBHHBx'
FCGI_BeginRequestBody = '!HB5x'
FCGI_EndRequestBody = '!LB3x'
FCGI_UnknownTypeBody = '!B7x'
FCGI_BeginRequestBody_LEN = struct.calcsize(FCGI_BeginRequestBody)
FCGI_EndRequestBody_LEN = struct.calcsize(FCGI_EndRequestBody)
FCGI_UnknownTypeBody_LEN = struct.calcsize(FCGI_UnknownTypeBody)
if __debug__:
    import time
    DEBUG = 0
    DEBUGLOG = '/www/server/mdserver-web/logs/fastcgi.log'

    def _debug(level, msg):
        if False:
            while True:
                i = 10
        if DEBUG < level:
            return
        try:
            f = open(DEBUGLOG, 'a')
            f.write('%sfcgi: %s\n' % (time.ctime()[4:-4], msg))
            f.close()
        except:
            pass

def decode_pair(s, pos=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decodes a name/value pair.\n\n    The number of bytes decoded as well as the name/value pair\n    are returned.\n    '
    nameLength = ord(s[pos])
    if nameLength & 128:
        nameLength = struct.unpack('!L', s[pos:pos + 4])[0] & 2147483647
        pos += 4
    else:
        pos += 1
    valueLength = ord(s[pos])
    if valueLength & 128:
        valueLength = struct.unpack('!L', s[pos:pos + 4])[0] & 2147483647
        pos += 4
    else:
        pos += 1
    name = s[pos:pos + nameLength]
    pos += nameLength
    value = s[pos:pos + valueLength]
    pos += valueLength
    return (pos, (name, value))

def encode_pair(name, value):
    if False:
        print('Hello World!')
    '\n    Encodes a name/value pair.\n\n    The encoded string is returned.\n    '
    nameLength = len(name)
    if nameLength < 128:
        s = chr(nameLength).encode()
    else:
        s = struct.pack('!L', nameLength | 2147483648)
    valueLength = len(value)
    if valueLength < 128:
        s += chr(valueLength).encode()
    else:
        s += struct.pack('!L', valueLength | 2147483648)
    return s + name + value

class Record(object):
    """
    A FastCGI Record.

    Used for encoding/decoding records.
    """

    def __init__(self, type=FCGI_UNKNOWN_TYPE, requestId=FCGI_NULL_REQUEST_ID):
        if False:
            return 10
        self.version = FCGI_VERSION_1
        self.type = type
        self.requestId = requestId
        self.contentLength = 0
        self.paddingLength = 0
        self.contentData = ''

    def _recvall(sock, length):
        if False:
            while True:
                i = 10
        '\n        Attempts to receive length bytes from a socket, blocking if necessary.\n        (Socket may be blocking or non-blocking.)\n        '
        dataList = []
        recvLen = 0
        while length:
            try:
                data = sock.recv(length)
            except socket.error as e:
                if e[0] == errno.EAGAIN:
                    select.select([sock], [], [])
                    continue
                else:
                    raise
            if not data:
                break
            dataList.append(data)
            dataLen = len(data)
            recvLen += dataLen
            length -= dataLen
        return (b''.join(dataList), recvLen)
    _recvall = staticmethod(_recvall)

    def read(self, sock):
        if False:
            print('Hello World!')
        'Read and decode a Record from a socket.'
        try:
            (header, length) = self._recvall(sock, FCGI_HEADER_LEN)
        except:
            raise EOFError
        if length < FCGI_HEADER_LEN:
            raise EOFError
        (self.version, self.type, self.requestId, self.contentLength, self.paddingLength) = struct.unpack(FCGI_Header, header)
        if __debug__:
            _debug(9, 'read: fd = %d, type = %d, requestId = %d, contentLength = %d' % (sock.fileno(), self.type, self.requestId, self.contentLength))
        if self.contentLength:
            try:
                (self.contentData, length) = self._recvall(sock, self.contentLength)
            except:
                raise EOFError
            if length < self.contentLength:
                raise EOFError
        if self.paddingLength:
            try:
                self._recvall(sock, self.paddingLength)
            except:
                raise EOFError

    def _sendall(sock, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Writes data to a socket and does not return until all the data is sent.\n        '
        length = len(data)
        while length:
            try:
                sent = sock.send(data)
            except socket.error as e:
                if e[0] == errno.EAGAIN:
                    select.select([], [sock], [])
                    continue
                else:
                    raise
            data = data[sent:]
            length -= sent
    _sendall = staticmethod(_sendall)

    def write(self, sock):
        if False:
            for i in range(10):
                print('nop')
        'Encode and write a Record to a socket.'
        self.paddingLength = -self.contentLength & 7
        if __debug__:
            _debug(9, 'write: fd = %d, type = %d, requestId = %d, contentLength = %d' % (sock.fileno(), self.type, self.requestId, self.contentLength))
        header = struct.pack(FCGI_Header, self.version, self.type, self.requestId, self.contentLength, self.paddingLength)
        self._sendall(sock, header)
        if self.contentLength:
            self._sendall(sock, self.contentData)
        if self.paddingLength:
            self._sendall(sock, b'\x00' * self.paddingLength)

class FCGIApp(object):

    def __init__(self, connect=None, host=None, port=None, filterEnviron=True):
        if False:
            return 10
        if host is not None:
            assert port is not None
            connect = (host, port)
        self._connect = connect
        self._filterEnviron = filterEnviron

    def __call__(self, environ, io, start_response=None):
        if False:
            print('Hello World!')
        sock = self._getConnection()
        requestId = 1
        rec = Record(FCGI_BEGIN_REQUEST, requestId)
        rec.contentData = struct.pack(FCGI_BeginRequestBody, FCGI_RESPONDER, 0)
        rec.contentLength = FCGI_BeginRequestBody_LEN
        rec.write(sock)
        if self._filterEnviron:
            params = self._defaultFilterEnviron(environ)
        else:
            params = self._lightFilterEnviron(environ)
        self._fcgiParams(sock, requestId, params)
        self._fcgiParams(sock, requestId, {})
        content_length = int(environ.get('CONTENT_LENGTH') or 0)
        s = ''
        while True:
            if not io:
                break
            chunk_size = min(content_length, 4096)
            s = io.read(chunk_size)
            content_length -= len(s)
            rec = Record(FCGI_STDIN, requestId)
            rec.contentData = s
            rec.contentLength = len(s)
            rec.write(sock)
            if not s:
                break
        rec = Record(FCGI_DATA, requestId)
        rec.write(sock)
        return sock

    def _getConnection(self):
        if False:
            for i in range(10):
                print('nop')
        if self._connect is not None:
            if isinstance(self._connect, str):
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(self._connect)
            elif hasattr(socket, 'create_connection'):
                sock = socket.create_connection(self._connect)
            else:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(self._connect)
            return sock
        raise NotImplementedError

    def _fcgiGetValues(self, sock, vars):
        if False:
            i = 10
            return i + 15
        outrec = Record(FCGI_GET_VALUES)
        data = []
        for name in vars:
            data.append(encode_pair(name, ''))
        data = ''.join(data)
        outrec.contentData = data
        outrec.contentLength = len(data)
        outrec.write(sock)
        inrec = Record()
        inrec.read(sock)
        result = {}
        if inrec.type == FCGI_GET_VALUES_RESULT:
            pos = 0
            while pos < inrec.contentLength:
                (pos, (name, value)) = decode_pair(inrec.contentData, pos)
                result[name] = value
        return result

    def _fcgiParams(self, sock, requestId, params):
        if False:
            while True:
                i = 10
        rec = Record(FCGI_PARAMS, requestId)
        data = []
        for (name, value) in params.items():
            data.append(encode_pair(name.encode('latin-1'), value.encode('latin-1')))
        data = b''.join(data)
        rec.contentData = data
        rec.contentLength = len(data)
        rec.write(sock)
    _environPrefixes = ['SERVER_', 'HTTP_', 'REQUEST_', 'REMOTE_', 'PATH_', 'CONTENT_', 'DOCUMENT_', 'SCRIPT_']
    _environCopies = ['SCRIPT_NAME', 'QUERY_STRING', 'AUTH_TYPE']
    _environRenames = []

    def _defaultFilterEnviron(self, environ):
        if False:
            return 10
        result = {}
        for n in environ.keys():
            iv = False
            for p in self._environPrefixes:
                if n.startswith(p):
                    result[n] = environ[n]
                    iv = True
            if n in self._environCopies:
                result[n] = environ[n]
                iv = True
            if n in self._environRenames:
                result[self._environRenames[n]] = environ[n]
                iv = True
            if not iv:
                result[n] = environ[n]
        return result

    def _lightFilterEnviron(self, environ):
        if False:
            print('Hello World!')
        result = {}
        for n in environ.keys():
            if n.upper() == n:
                result[n] = environ[n]
        return result