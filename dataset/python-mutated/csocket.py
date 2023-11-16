__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
from pickle import dumps, loads, HIGHEST_PROTOCOL
from ..protocol import CSPROTO
import socket
import sys

class CSocket:

    def __init__(self, sock='/var/run/fail2ban/fail2ban.sock', timeout=-1):
        if False:
            return 10
        self.__csock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.__deftout = self.__csock.gettimeout()
        if timeout != -1:
            self.settimeout(timeout)
        self.__csock.connect(sock)

    def __del__(self):
        if False:
            while True:
                i = 10
        self.close()

    def send(self, msg, nonblocking=False, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        obj = dumps(list(map(CSocket.convert, msg)), HIGHEST_PROTOCOL)
        self.__csock.send(obj)
        self.__csock.send(CSPROTO.END)
        return self.receive(self.__csock, nonblocking, timeout)

    def settimeout(self, timeout):
        if False:
            return 10
        self.__csock.settimeout(timeout if timeout != -1 else self.__deftout)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.__csock:
            return
        try:
            self.__csock.sendall(CSPROTO.CLOSE + CSPROTO.END)
            self.__csock.shutdown(socket.SHUT_RDWR)
        except socket.error:
            pass
        try:
            self.__csock.close()
        except socket.error:
            pass
        self.__csock = None

    @staticmethod
    def convert(m):
        if False:
            print('Hello World!')
        'Convert every "unexpected" member of message to string'
        if isinstance(m, (str, bool, int, float, list, dict, set)):
            return m
        else:
            return str(m)

    @staticmethod
    def receive(sock, nonblocking=False, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        msg = CSPROTO.EMPTY
        if nonblocking:
            sock.setblocking(0)
        if timeout:
            sock.settimeout(timeout)
        bufsize = 1024
        while msg.rfind(CSPROTO.END, -32) == -1:
            chunk = sock.recv(bufsize)
            if not len(chunk):
                raise socket.error(104, 'Connection reset by peer')
            if chunk == CSPROTO.END:
                break
            msg = msg + chunk
            if bufsize < 32768:
                bufsize <<= 1
        return loads(msg)