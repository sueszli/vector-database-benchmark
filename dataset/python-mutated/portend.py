"""
A simple library for managing the availability of ports.
"""
import time
import socket
import argparse
import sys
import itertools
import contextlib
import platform
from collections import abc
import urllib.parse
from tempora import timing

def client_host(server_host):
    if False:
        return 10
    "\n    Return the host on which a client can connect to the given listener.\n\n    >>> client_host('192.168.0.1')\n    '192.168.0.1'\n    >>> client_host('0.0.0.0')\n    '127.0.0.1'\n    >>> client_host('::')\n    '::1'\n    "
    if server_host == '0.0.0.0':
        return '127.0.0.1'
    if server_host in ('::', '::0', '::0.0.0.0'):
        return '::1'
    return server_host

class Checker(object):

    def __init__(self, timeout=1.0):
        if False:
            i = 10
            return i + 15
        self.timeout = timeout

    def assert_free(self, host, port=None):
        if False:
            return 10
        "\n        Assert that the given addr is free\n        in that all attempts to connect fail within the timeout\n        or raise a PortNotFree exception.\n\n        >>> free_port = find_available_local_port()\n\n        >>> Checker().assert_free('localhost', free_port)\n        >>> Checker().assert_free('127.0.0.1', free_port)\n        >>> Checker().assert_free('::1', free_port)\n\n        Also accepts an addr tuple\n\n        >>> addr = '::1', free_port, 0, 0\n        >>> Checker().assert_free(addr)\n\n        Host might refer to a server bind address like '::', which\n        should use localhost to perform the check.\n\n        >>> Checker().assert_free('::', free_port)\n        "
        if port is None and isinstance(host, abc.Sequence):
            (host, port) = host[:2]
        if platform.system() == 'Windows':
            host = client_host(host)
        info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        list(itertools.starmap(self._connect, info))

    def _connect(self, af, socktype, proto, canonname, sa):
        if False:
            return 10
        s = socket.socket(af, socktype, proto)
        s.settimeout(self.timeout)
        with contextlib.closing(s):
            try:
                s.connect(sa)
            except socket.error:
                return
        (host, port) = sa[:2]
        tmpl = 'Port {port} is in use on {host}.'
        raise PortNotFree(tmpl.format(**locals()))

class Timeout(IOError):
    pass

class PortNotFree(IOError):
    pass

def free(host, port, timeout=float('Inf')):
    if False:
        i = 10
        return i + 15
    "\n    Wait for the specified port to become free (dropping or rejecting\n    requests). Return when the port is free or raise a Timeout if timeout has\n    elapsed.\n\n    Timeout may be specified in seconds or as a timedelta.\n    If timeout is None or ∞, the routine will run indefinitely.\n\n    >>> free('localhost', find_available_local_port())\n\n    >>> free(None, None)\n    Traceback (most recent call last):\n    ...\n    ValueError: Host values of '' or None are not allowed.\n    "
    if not host:
        raise ValueError("Host values of '' or None are not allowed.")
    timer = timing.Timer(timeout)
    while True:
        try:
            Checker(timeout=0.1).assert_free(host, port)
            return
        except PortNotFree:
            if timer.expired():
                raise Timeout('Port {port} not free on {host}.'.format(**locals()))
            time.sleep(0.1)

def occupied(host, port, timeout=float('Inf')):
    if False:
        return 10
    "\n    Wait for the specified port to become occupied (accepting requests).\n    Return when the port is occupied or raise a Timeout if timeout has\n    elapsed.\n\n    Timeout may be specified in seconds or as a timedelta.\n    If timeout is None or ∞, the routine will run indefinitely.\n\n    >>> occupied('localhost', find_available_local_port(), .1)\n    Traceback (most recent call last):\n    ...\n    Timeout: Port ... not bound on localhost.\n\n    >>> occupied(None, None)\n    Traceback (most recent call last):\n    ...\n    ValueError: Host values of '' or None are not allowed.\n    "
    if not host:
        raise ValueError("Host values of '' or None are not allowed.")
    timer = timing.Timer(timeout)
    while True:
        try:
            Checker(timeout=0.5).assert_free(host, port)
            if timer.expired():
                raise Timeout('Port {port} not bound on {host}.'.format(**locals()))
            time.sleep(0.1)
        except PortNotFree:
            return

def find_available_local_port():
    if False:
        print('Hello World!')
    '\n    Find a free port on localhost.\n\n    >>> 0 < find_available_local_port() < 65536\n    True\n    '
    infos = socket.getaddrinfo(None, 0, socket.AF_UNSPEC, socket.SOCK_STREAM)
    (family, proto, _, _, addr) = next(iter(infos))
    sock = socket.socket(family, proto)
    sock.bind(addr)
    (addr, port) = sock.getsockname()[:2]
    sock.close()
    return port

class HostPort(str):
    """
    A simple representation of a host/port pair as a string

    >>> hp = HostPort('localhost:32768')

    >>> hp.host
    'localhost'

    >>> hp.port
    32768

    >>> len(hp)
    15

    >>> hp = HostPort('[::1]:32768')

    >>> hp.host
    '::1'

    >>> hp.port
    32768
    """

    @property
    def host(self):
        if False:
            i = 10
            return i + 15
        return urllib.parse.urlparse(f'//{self}').hostname

    @property
    def port(self):
        if False:
            while True:
                i = 10
        return urllib.parse.urlparse(f'//{self}').port

    @classmethod
    def from_addr(cls, addr):
        if False:
            i = 10
            return i + 15
        (listen_host, port) = addr[:2]
        plain_host = client_host(listen_host)
        host = f'[{plain_host}]' if ':' in plain_host else plain_host
        return cls(':'.join([host, str(port)]))

def _main(args=None):
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()

    def global_lookup(key):
        if False:
            return 10
        return globals()[key]
    parser.add_argument('target', metavar='host:port', type=HostPort)
    parser.add_argument('func', metavar='state', type=global_lookup)
    parser.add_argument('-t', '--timeout', default=None, type=float)
    args = parser.parse_args(args)
    try:
        args.func(args.target.host, args.target.port, timeout=args.timeout)
    except Timeout as timeout:
        print(timeout, file=sys.stderr)
        raise SystemExit(1)
__name__ == '__main__' and _main()