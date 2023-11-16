"""Mock socket module used by the smtpd and smtplib tests.
"""
import socket as socket_module
_defaulttimeout = None
_reply_data = None

def reply_with(line):
    if False:
        return 10
    global _reply_data
    _reply_data = line

class MockFile:
    """Mock file object returned by MockSocket.makefile().
    """

    def __init__(self, lines):
        if False:
            print('Hello World!')
        self.lines = lines

    def readline(self, limit=-1):
        if False:
            print('Hello World!')
        result = self.lines.pop(0) + b'\r\n'
        if limit >= 0:
            self.lines.insert(0, result[limit:-2])
            result = result[:limit]
        return result

    def close(self):
        if False:
            print('Hello World!')
        pass

class MockSocket:
    """Mock socket object used by smtpd and smtplib tests.
    """

    def __init__(self, family=None):
        if False:
            print('Hello World!')
        global _reply_data
        self.family = family
        self.output = []
        self.lines = []
        if _reply_data:
            self.lines.append(_reply_data)
            _reply_data = None
        self.conn = None
        self.timeout = None

    def queue_recv(self, line):
        if False:
            for i in range(10):
                print('nop')
        self.lines.append(line)

    def recv(self, bufsize, flags=None):
        if False:
            i = 10
            return i + 15
        data = self.lines.pop(0) + b'\r\n'
        return data

    def fileno(self):
        if False:
            while True:
                i = 10
        return 0

    def settimeout(self, timeout):
        if False:
            i = 10
            return i + 15
        if timeout is None:
            self.timeout = _defaulttimeout
        else:
            self.timeout = timeout

    def gettimeout(self):
        if False:
            for i in range(10):
                print('nop')
        return self.timeout

    def setsockopt(self, level, optname, value):
        if False:
            print('Hello World!')
        pass

    def getsockopt(self, level, optname, buflen=None):
        if False:
            print('Hello World!')
        return 0

    def bind(self, address):
        if False:
            return 10
        pass

    def accept(self):
        if False:
            i = 10
            return i + 15
        self.conn = MockSocket()
        return (self.conn, 'c')

    def getsockname(self):
        if False:
            for i in range(10):
                print('nop')
        return ('0.0.0.0', 0)

    def setblocking(self, flag):
        if False:
            while True:
                i = 10
        pass

    def listen(self, backlog):
        if False:
            for i in range(10):
                print('nop')
        pass

    def makefile(self, mode='r', bufsize=-1):
        if False:
            return 10
        handle = MockFile(self.lines)
        return handle

    def sendall(self, data, flags=None):
        if False:
            print('Hello World!')
        self.last = data
        self.output.append(data)
        return len(data)

    def send(self, data, flags=None):
        if False:
            print('Hello World!')
        self.last = data
        self.output.append(data)
        return len(data)

    def getpeername(self):
        if False:
            i = 10
            return i + 15
        return ('peer-address', 'peer-port')

    def close(self):
        if False:
            while True:
                i = 10
        pass

    def connect(self, host):
        if False:
            while True:
                i = 10
        pass

def socket(family=None, type=None, proto=None):
    if False:
        i = 10
        return i + 15
    return MockSocket(family)

def create_connection(address, timeout=socket_module._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    if False:
        for i in range(10):
            print('nop')
    try:
        int_port = int(address[1])
    except ValueError:
        raise error
    ms = MockSocket()
    if timeout is socket_module._GLOBAL_DEFAULT_TIMEOUT:
        timeout = getdefaulttimeout()
    ms.settimeout(timeout)
    return ms

def setdefaulttimeout(timeout):
    if False:
        while True:
            i = 10
    global _defaulttimeout
    _defaulttimeout = timeout

def getdefaulttimeout():
    if False:
        for i in range(10):
            print('nop')
    return _defaulttimeout

def getfqdn():
    if False:
        for i in range(10):
            print('nop')
    return ''

def gethostname():
    if False:
        i = 10
        return i + 15
    pass

def gethostbyname(name):
    if False:
        i = 10
        return i + 15
    return ''

def getaddrinfo(*args, **kw):
    if False:
        while True:
            i = 10
    return socket_module.getaddrinfo(*args, **kw)
gaierror = socket_module.gaierror
error = socket_module.error
_GLOBAL_DEFAULT_TIMEOUT = socket_module._GLOBAL_DEFAULT_TIMEOUT
AF_INET = socket_module.AF_INET
AF_INET6 = socket_module.AF_INET6
SOCK_STREAM = socket_module.SOCK_STREAM
SOL_SOCKET = None
SO_REUSEADDR = None
if hasattr(socket_module, 'AF_UNIX'):
    AF_UNIX = socket_module.AF_UNIX