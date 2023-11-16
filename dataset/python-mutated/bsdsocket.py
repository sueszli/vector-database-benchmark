"""BSD socket interface communication utilities"""
import os
import pickle
import socket
import struct
import threading
import errno
import traceback
from spyder.config.base import get_debug_level, STDERR
DEBUG_EDITOR = get_debug_level() >= 3
PICKLE_HIGHEST_PROTOCOL = 2

def temp_fail_retry(error, fun, *args):
    if False:
        return 10
    'Retry to execute function, ignoring EINTR error (interruptions)'
    while 1:
        try:
            return fun(*args)
        except error as e:
            eintr = errno.WSAEINTR if os.name == 'nt' else errno.EINTR
            if e.args[0] == eintr:
                continue
            raise
SZ = struct.calcsize('l')

def write_packet(sock, data, already_pickled=False):
    if False:
        return 10
    'Write *data* to socket *sock*'
    if already_pickled:
        sent_data = data
    else:
        sent_data = pickle.dumps(data, PICKLE_HIGHEST_PROTOCOL)
    sent_data = struct.pack('l', len(sent_data)) + sent_data
    nsend = len(sent_data)
    while nsend > 0:
        nsend -= temp_fail_retry(socket.error, sock.send, sent_data)

def read_packet(sock, timeout=None):
    if False:
        while True:
            i = 10
    '\n    Read data from socket *sock*\n    Returns None if something went wrong\n    '
    sock.settimeout(timeout)
    (dlen, data) = (None, None)
    try:
        if os.name == 'nt':
            datalen = sock.recv(SZ)
            (dlen,) = struct.unpack('l', datalen)
            data = b''
            while len(data) < dlen:
                data += sock.recv(dlen)
        else:
            datalen = temp_fail_retry(socket.error, sock.recv, SZ, socket.MSG_WAITALL)
            if len(datalen) == SZ:
                (dlen,) = struct.unpack('l', datalen)
                data = temp_fail_retry(socket.error, sock.recv, dlen, socket.MSG_WAITALL)
    except socket.timeout:
        raise
    except socket.error:
        data = None
    finally:
        sock.settimeout(None)
    if data is not None:
        try:
            return pickle.loads(data)
        except Exception:
            if DEBUG_EDITOR:
                traceback.print_exc(file=STDERR)
            return
COMMUNICATE_LOCK = threading.Lock()

def communicate(sock, command, settings=[]):
    if False:
        print('Hello World!')
    'Communicate with monitor'
    try:
        COMMUNICATE_LOCK.acquire()
        write_packet(sock, command)
        for option in settings:
            write_packet(sock, option)
        return read_packet(sock)
    finally:
        COMMUNICATE_LOCK.release()

class PacketNotReceived(object):
    pass
PACKET_NOT_RECEIVED = PacketNotReceived()
if __name__ == '__main__':
    if not os.name == 'nt':
        print('-- Testing standard Python socket interface --')
        address = ('127.0.0.1', 9999)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setblocking(0)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(address)
        server.listen(2)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(address)
        client.send('data to be catched'.encode('utf-8'))
        (accsock, addr) = server.accept()
        print('..got "%s" from %s' % (accsock.recv(4096), addr))
        print('-- Testing BSD socket write_packet/read_packet --')
        write_packet(client, 'a tiny piece of data')
        print('..got "%s" from read_packet()' % read_packet(accsock))
        client.close()
        server.close()
        print('-- Done.')