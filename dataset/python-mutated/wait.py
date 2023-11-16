from __future__ import annotations
import select
import socket
from functools import partial
__all__ = ['wait_for_read', 'wait_for_write']

def select_wait_for_socket(sock: socket.socket, read: bool=False, write: bool=False, timeout: float | None=None) -> bool:
    if False:
        while True:
            i = 10
    if not read and (not write):
        raise RuntimeError('must specify at least one of read=True, write=True')
    rcheck = []
    wcheck = []
    if read:
        rcheck.append(sock)
    if write:
        wcheck.append(sock)
    fn = partial(select.select, rcheck, wcheck, wcheck)
    (rready, wready, xready) = fn(timeout)
    return bool(rready or wready or xready)

def poll_wait_for_socket(sock: socket.socket, read: bool=False, write: bool=False, timeout: float | None=None) -> bool:
    if False:
        i = 10
        return i + 15
    if not read and (not write):
        raise RuntimeError('must specify at least one of read=True, write=True')
    mask = 0
    if read:
        mask |= select.POLLIN
    if write:
        mask |= select.POLLOUT
    poll_obj = select.poll()
    poll_obj.register(sock, mask)

    def do_poll(t: float | None) -> list[tuple[int, int]]:
        if False:
            while True:
                i = 10
        if t is not None:
            t *= 1000
        return poll_obj.poll(t)
    return bool(do_poll(timeout))

def _have_working_poll() -> bool:
    if False:
        i = 10
        return i + 15
    try:
        poll_obj = select.poll()
        poll_obj.poll(0)
    except (AttributeError, OSError):
        return False
    else:
        return True

def wait_for_socket(sock: socket.socket, read: bool=False, write: bool=False, timeout: float | None=None) -> bool:
    if False:
        while True:
            i = 10
    global wait_for_socket
    if _have_working_poll():
        wait_for_socket = poll_wait_for_socket
    elif hasattr(select, 'select'):
        wait_for_socket = select_wait_for_socket
    return wait_for_socket(sock, read, write, timeout)

def wait_for_read(sock: socket.socket, timeout: float | None=None) -> bool:
    if False:
        print('Hello World!')
    'Waits for reading to be available on a given socket.\n    Returns True if the socket is readable, or False if the timeout expired.\n    '
    return wait_for_socket(sock, read=True, timeout=timeout)

def wait_for_write(sock: socket.socket, timeout: float | None=None) -> bool:
    if False:
        print('Hello World!')
    'Waits for writing to be available on a given socket.\n    Returns True if the socket is readable, or False if the timeout expired.\n    '
    return wait_for_socket(sock, write=True, timeout=timeout)