import socket
import logging
logger = logging.getLogger('jade.tcp')

class JadeTCPImpl:
    PROTOCOL_PREFIX = 'tcp:'

    @classmethod
    def isSupportedDevice(cls, device):
        if False:
            for i in range(10):
                print('nop')
        return device is not None and device.startswith(cls.PROTOCOL_PREFIX)

    def __init__(self, device):
        if False:
            while True:
                i = 10
        assert self.isSupportedDevice(device)
        self.device = device
        self.tcp_sock = None

    def connect(self):
        if False:
            i = 10
            return i + 15
        assert self.isSupportedDevice(self.device)
        assert self.tcp_sock is None
        logger.info('Connecting to {}'.format(self.device))
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        url = self.device[len(self.PROTOCOL_PREFIX):].split(':')
        self.tcp_sock.connect((url[0], int(url[1])))
        assert self.tcp_sock is not None
        self.tcp_sock.__enter__()
        logger.info('Connected')

    def disconnect(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.tcp_sock is not None
        self.tcp_sock.__exit__()
        self.tcp_sock = None

    def write(self, bytes_):
        if False:
            i = 10
            return i + 15
        assert self.tcp_sock is not None
        return self.tcp_sock.send(bytes_)

    def read(self, n):
        if False:
            i = 10
            return i + 15
        assert self.tcp_sock is not None
        return self.tcp_sock.recv(n)