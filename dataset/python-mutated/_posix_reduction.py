import os
import socket
import _socket
from multiprocessing.connection import Connection
from multiprocessing.context import get_spawning_popen
from .reduction import register
HAVE_SEND_HANDLE = hasattr(socket, 'CMSG_LEN') and hasattr(socket, 'SCM_RIGHTS') and hasattr(socket.socket, 'sendmsg')

def _mk_inheritable(fd):
    if False:
        i = 10
        return i + 15
    os.set_inheritable(fd, True)
    return fd

def DupFd(fd):
    if False:
        while True:
            i = 10
    'Return a wrapper for an fd.'
    popen_obj = get_spawning_popen()
    if popen_obj is not None:
        return popen_obj.DupFd(popen_obj.duplicate_for_child(fd))
    elif HAVE_SEND_HANDLE:
        from multiprocessing import resource_sharer
        return resource_sharer.DupFd(fd)
    else:
        raise TypeError('Cannot pickle connection object. This object can only be passed when spawning a new process')

def _reduce_socket(s):
    if False:
        print('Hello World!')
    df = DupFd(s.fileno())
    return (_rebuild_socket, (df, s.family, s.type, s.proto))

def _rebuild_socket(df, family, type, proto):
    if False:
        print('Hello World!')
    fd = df.detach()
    return socket.fromfd(fd, family, type, proto)

def rebuild_connection(df, readable, writable):
    if False:
        while True:
            i = 10
    fd = df.detach()
    return Connection(fd, readable, writable)

def reduce_connection(conn):
    if False:
        for i in range(10):
            print('nop')
    df = DupFd(conn.fileno())
    return (rebuild_connection, (df, conn.readable, conn.writable))
register(socket.socket, _reduce_socket)
register(_socket.socket, _reduce_socket)
register(Connection, reduce_connection)