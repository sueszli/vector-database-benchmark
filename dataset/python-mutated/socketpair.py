from __future__ import absolute_import
import sys
import socket
import errno
_LOCALHOST = '127.0.0.1'
_LOCALHOST_V6 = '::1'
if not hasattr(socket, 'socketpair'):

    def socketpair(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0):
        if False:
            while True:
                i = 10
        if family == socket.AF_INET:
            host = _LOCALHOST
        elif family == socket.AF_INET6:
            host = _LOCALHOST_V6
        else:
            raise ValueError('Only AF_INET and AF_INET6 socket address families are supported')
        if type != socket.SOCK_STREAM:
            raise ValueError('Only SOCK_STREAM socket type is supported')
        if proto != 0:
            raise ValueError('Only protocol zero is supported')
        lsock = socket.socket(family, type, proto)
        try:
            lsock.bind((host, 0))
            lsock.listen(min(socket.SOMAXCONN, 128))
            (addr, port) = lsock.getsockname()[:2]
            csock = socket.socket(family, type, proto)
            try:
                csock.setblocking(False)
                if sys.version_info >= (3, 0):
                    try:
                        csock.connect((addr, port))
                    except (BlockingIOError, InterruptedError):
                        pass
                else:
                    try:
                        csock.connect((addr, port))
                    except socket.error as e:
                        if e.errno != errno.WSAEWOULDBLOCK:
                            raise
                csock.setblocking(True)
                (ssock, _) = lsock.accept()
            except Exception:
                csock.close()
                raise
        finally:
            lsock.close()
        return (ssock, csock)
    socket.socketpair = socketpair