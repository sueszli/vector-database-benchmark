import ipaddress
import logging
import re
import uuid
from functools import partial
from typing import Callable, List, Optional
from golem.core.types import Kwargs
from golem.core import variables
logger = logging.getLogger(__name__)

class SocketAddress:
    """TCP socket address (host and port)"""
    _dns_label_pattern = re.compile('(?!-)[a-z\\d-]{1,63}(?<!-)\\Z', re.IGNORECASE)
    _all_numeric_pattern = re.compile('[0-9\\.]+\\Z')

    @classmethod
    def is_proper_address(cls, address, port):
        if False:
            print('Hello World!')
        try:
            SocketAddress(address, port)
        except Exception as exc:
            logger.info('Wrong address %r', exc)
            return False
        return True

    def __init__(self, address, port):
        if False:
            i = 10
            return i + 15
        "Creates and validates SocketAddress. Raises\n        AddressValueError if 'address' or 'port' is invalid.\n        :param str address: IPv4/IPv6 address or hostname\n        :param int port:\n        "
        self.address = address
        self.port = port
        self.ipv6 = False
        self.hostname = False
        try:
            self.__validate()
        except ValueError as err:
            raise ipaddress.AddressValueError(err)

    def __validate(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.address, str):
            self.address = self.address
        else:
            raise TypeError('Address must be a string, not a ' + type(self.address).__name__)
        if not isinstance(self.port, int):
            raise TypeError('Port must be an int, not a ' + type(self.port).__name__)
        if self.address.find(':') != -1:
            if self.address.find('%') != -1:
                self.address = self.address[:self.address.find('%')]
            ipaddress.IPv6Address(self.address)
            self.ipv6 = True
        elif self._all_numeric_pattern.match(self.address):
            ipaddress.IPv4Address(self.address)
        else:
            SocketAddress.validate_hostname(self.address)
            self.hostname = True
        if not variables.MIN_PORT <= self.port <= variables.MAX_PORT:
            raise ValueError('Port out of range ({} .. {}): {}'.format(variables.MIN_PORT, variables.MAX_PORT, self.port))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return self.address == other.address and self.port == other.port

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'SocketAddress(%r, %r)' % (self.address, self.port)

    def __str__(self):
        if False:
            print('Hello World!')
        return self.address + ':' + str(self.port)

    @staticmethod
    def validate_hostname(hostname):
        if False:
            return 10
        'Checks that the given string is a valid hostname.\n        See RFC 1123, page 13, and here:\n        http://stackoverflow.com/questions/2532053/validate-a-hostname-string.\n        Raises ValueError if the argument is not a valid hostname.\n        :param str hostname:\n        :returns None\n        '
        if not isinstance(hostname, str):
            raise TypeError('Expected string argument, not ' + type(hostname).__name__)
        if hostname == '':
            raise ValueError('Empty host name')
        if len(hostname) > 255:
            raise ValueError('Host name exceeds 255 chars: ' + hostname)
        if hostname.endswith('.'):
            hostname = hostname[:-1]
        segments = hostname.split('.')
        if not all((SocketAddress._dns_label_pattern.match(s) for s in segments)):
            raise ValueError('Invalid host name: ' + hostname)

    @staticmethod
    def parse(string):
        if False:
            return 10
        "Parses a string representation of a socket address.\n        IPv4 syntax: <IPv4 address> ':' <port>\n        IPv6 syntax: '[' <IPv6 address> ']' ':' <port>\n        DNS syntax:  <hostname> ':' <port>\n        Raises AddressValueError if the input cannot be parsed.\n        :param str string:\n        :returns parsed SocketAddress\n        :rtype SocketAddress\n        "
        if not isinstance(string, str):
            raise TypeError('Expected string argument, not ' + type(string).__name__)
        try:
            if string.startswith('['):
                (addr_str, port_str) = string.split(']:')
                addr_str = addr_str[1:]
            else:
                (addr_str, port_str) = string.split(':')
            port = int(port_str)
        except ValueError:
            raise ipaddress.AddressValueError('Invalid address "{}"'.format(string))
        return SocketAddress(addr_str, port)

class TCPListenInfo(object):

    def __init__(self, port_start: int, port_end: Optional[int]=None, established_callback: Optional[Callable]=None, failure_callback: Optional[Callable]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Information needed for listen function. Network will try to start\n        listening on port_start, then iterate by 1 to port_end.\n        If port_end is None, than network will only try to listen on\n        port_start.\n        :param port_start: try to start listening from that port\n        :param port_end: *Default: None* highest port that network will try to\n                         listen on\n        :param established_callback: *Default: None* deferred callback after\n                                     listening established\n        :param failure_callback: *Default: None* deferred callback after\n                                 listening failure\n        :return:\n        '
        self.port_start = port_start
        self.port_end = port_end if port_end else port_start
        self.established_callback = established_callback
        self.failure_callback = failure_callback

    def __str__(self):
        if False:
            return 10
        return 'TCP listen info: ports [{}:{}],callback: {}, errback: {}'.format(self.port_start, self.port_end, self.established_callback, self.failure_callback)

class TCPListeningInfo(object):

    def __init__(self, port: int, stopped_callback: Optional[Callable]=None, stopped_errback: Optional[Callable]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        TCP listening port information\n        :param port: port opened for listening\n        :param stopped_callback: *Default: None* deferred callback after\n                                 listening on this port is stopped\n        :param stopped_errback: *Default: None* deferred callback after stop\n                                listening failed\n        '
        self.port = port
        self.stopped_callback = stopped_callback
        self.stopped_errback = stopped_errback

    def __str__(self):
        if False:
            print('Hello World!')
        return 'A listening port {} information'.format(self.port)

class TCPConnectInfo(object):

    def __init__(self, socket_addresses: List[SocketAddress], established_callback: Optional[Callable]=None, failure_callback: Optional[Callable]=None, final_failure_callback: Optional[Callable]=None, kwargs: Kwargs=dict()) -> None:
        if False:
            print('Hello World!')
        '\n        Information for TCP connect function\n        '
        self.id = str(uuid.uuid4())
        self.socket_addresses = socket_addresses
        self.established_callback = partial(established_callback, conn_id=self.id, **kwargs) if established_callback else None
        self.failure_callback = partial(failure_callback, conn_id=self.id, **kwargs) if failure_callback else None
        self.final_failure_callback = partial(final_failure_callback, conn_id=self.id, **kwargs) if final_failure_callback else None

    def __str__(self):
        if False:
            i = 10
            return i + 15

        def get_func(cbk):
            if False:
                i = 10
                return i + 15
            return cbk.func if cbk is not None else None
        return 'TCP connection information: addresses {}, callback {}, errback {}, final_errback {}'.format(self.socket_addresses, get_func(self.established_callback), get_func(self.failure_callback), get_func(self.final_failure_callback))