"""Helper methods to tune a TCP connection"""
import asyncio
import socket
from contextlib import suppress
from typing import Optional
__all__ = ('tcp_keepalive', 'tcp_nodelay')
if hasattr(socket, 'SO_KEEPALIVE'):

    def tcp_keepalive(transport: asyncio.Transport) -> None:
        if False:
            for i in range(10):
                print('nop')
        sock = transport.get_extra_info('socket')
        if sock is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
else:

    def tcp_keepalive(transport: asyncio.Transport) -> None:
        if False:
            return 10
        pass

def tcp_nodelay(transport: asyncio.Transport, value: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    sock = transport.get_extra_info('socket')
    if sock is None:
        return
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return
    value = bool(value)
    with suppress(OSError):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, value)