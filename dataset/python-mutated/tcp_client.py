import socket
from routersploit.core.exploit.exploit import Exploit
from routersploit.core.exploit.exploit import Protocol
from routersploit.core.exploit.option import OptBool
from routersploit.core.exploit.printer import print_status
from routersploit.core.exploit.printer import print_error
from routersploit.core.exploit.utils import is_ipv4
from routersploit.core.exploit.utils import is_ipv6
TCP_SOCKET_TIMEOUT = 8.0

class TCPCli(object):
    """ TCP Client provides methods to handle communication with TCP server """

    def __init__(self, tcp_target: str, tcp_port: int, verbosity: bool=False) -> None:
        if False:
            print('Hello World!')
        ' TCP client constructor\n\n        :param str tcp_target: target TCP server ip address\n        :param int tcp_port: target TCP server port\n        :param bool verbosity: display verbose output\n        :return None:\n        '
        self.tcp_target = tcp_target
        self.tcp_port = tcp_port
        self.verbosity = verbosity
        self.peer = '{}:{}'.format(self.tcp_target, self.tcp_port)
        if is_ipv4(self.tcp_target):
            self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif is_ipv6(self.tcp_target):
            self.tcp_client = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        else:
            print_error('Target address is not valid IPv4 nor IPv6 address', verbose=self.verbosity)
            return None
        self.tcp_client.settimeout(TCP_SOCKET_TIMEOUT)

    def connect(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ' Connect to TCP server\n\n        :return bool: True if connection was successful, False otherwise\n        '
        try:
            self.tcp_client.connect((self.tcp_target, self.tcp_port))
            print_status(self.peer, 'TCP Connection established', verbose=self.verbosity)
            return True
        except Exception as err:
            print_error(self.peer, 'TCP Error while connecting to the server', err, verbose=self.verbosity)
        return False

    def send(self, data: bytes) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ' Send data to TCP server\n\n        :param bytes data: data that should be sent to TCP server\n        :return bool: True if sending data was successful, False otherwise\n        '
        try:
            self.tcp_client.send(data)
            return True
        except Exception as err:
            print_error(self.peer, 'TCP Error while sending data', err, verbose=self.verbosity)
        return False

    def recv(self, num: int) -> bytes:
        if False:
            print('Hello World!')
        ' Receive data from TCP server\n\n        :param int num: number of bytes that should be received from the server\n        :return bytes: data that was received from the server\n        '
        try:
            response = self.tcp_client.recv(num)
            return response
        except Exception as err:
            print_error(self.peer, 'TCP Error while receiving data', err, verbose=self.verbosity)
        return None

    def recv_all(self, num: int) -> bytes:
        if False:
            while True:
                i = 10
        ' Receive all data sent by the server\n\n        :param int num: number of total bytes that should be received\n        :return bytes: data that was received from the server\n        '
        try:
            response = b''
            received = 0
            while received < num:
                tmp = self.tcp_client.recv(num - received)
                if tmp:
                    received += len(tmp)
                    response += tmp
                else:
                    break
            return response
        except Exception as err:
            print_error(self.peer, 'TCP Error while receiving all data', err, verbose=self.verbosity)
        return None

    def close(self) -> bool:
        if False:
            i = 10
            return i + 15
        ' Close connection to TCP server\n\n        :return bool: True if closing connection was successful, False otherwise\n        '
        try:
            self.tcp_client.close()
            return True
        except Exception as err:
            print_error(self.peer, 'TCP Error while closing tcp socket', err, verbose=self.verbosity)
        return False

class TCPClient(Exploit):
    """ TCP Client exploit """
    target_protocol = Protocol.TCP
    verbosity = OptBool(True, 'Enable verbose output: true/false')

    def tcp_create(self, target: str=None, port: int=None) -> TCPCli:
        if False:
            return 10
        ' Creates TCP client\n\n        :param str target: target TCP server ip address\n        :param int port: target TCP server port\n        :return TCPCli: TCP client object\n        '
        tcp_target = target if target else self.target
        tcp_port = port if port else self.port
        tcp_client = TCPCli(tcp_target, tcp_port, verbosity=self.verbosity)
        return tcp_client