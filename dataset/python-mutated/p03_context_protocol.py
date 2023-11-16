"""
Topic: 让对象支持上下文管理器
Desc : 
"""
from socket import socket, AF_INET, SOCK_STREAM

class LazyConnection:

    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        if False:
            print('Hello World!')
        self.address = address
        self.family = family
        self.type = type
        self.sock = None

    def __enter__(self):
        if False:
            return 10
        if self.sock is not None:
            raise RuntimeError('Already connected')
        self.sock = socket(self.family, self.type)
        self.sock.connect(self.address)
        return self.sock

    def __exit__(self, exc_ty, exc_val, tb):
        if False:
            i = 10
            return i + 15
        self.sock.close()
        self.sock = None
from functools import partial
conn = LazyConnection(('www.python.org', 80))
with conn as s:
    s.send(b'GET /index.html HTTP/1.0\r\n')
    s.send(b'Host: www.python.org\r\n')
    s.send(b'\r\n')
    resp = b''.join(iter(partial(s.recv, 8192), b''))
    print(resp)

class LazyConnection2:

    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        if False:
            return 10
        self.address = address
        self.family = family
        self.type = type
        self.connections = []

    def __enter__(self):
        if False:
            print('Hello World!')
        sock = socket(self.family, self.type)
        sock.connect(self.address)
        self.connections.append(sock)
        return sock

    def __exit__(self, exc_ty, exc_val, tb):
        if False:
            return 10
        self.connections.pop().close()
from functools import partial
conn = LazyConnection2(('www.python.org', 80))
with conn as s1:
    pass
    with conn as s2:
        pass