import socket
from contextlib import contextmanager

@contextmanager
def report_outcome(tagline):
    if False:
        i = 10
        return i + 15
    try:
        yield
    except OSError as exc:
        print(f'{tagline}: failed')
        print(f'    details: {exc!r}')
    else:
        print(f'{tagline}: succeeded')
lsock = socket.socket()
lsock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
lsock.bind(('127.0.0.1', 0))
sockaddr = lsock.getsockname()
lsock.listen(10)
csock = socket.socket()
csock.connect(sockaddr)
(ssock, _) = lsock.accept()
print('lsock', lsock.getsockname())
print('ssock', ssock.getsockname())
probe = socket.socket()
probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
with report_outcome('rebind with existing listening socket'):
    probe.bind(sockaddr)
lsock.close()
probe = socket.socket()
probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
with report_outcome('rebind with live connected sockets'):
    probe.bind(sockaddr)
    probe.listen(10)
    print('probe', probe.getsockname())
    print('ssock', ssock.getsockname())
probe.close()
ssock.send(b'x')
assert csock.recv(1) == b'x'
ssock.close()
assert csock.recv(1) == b''
probe = socket.socket()
probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
with report_outcome('rebind with TIME_WAIT socket'):
    probe.bind(sockaddr)
    probe.listen(10)
probe.close()