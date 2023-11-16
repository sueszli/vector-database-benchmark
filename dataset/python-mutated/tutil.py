from __future__ import annotations
import asyncio
import gc
import os
import socket as stdlib_socket
import sys
import warnings
from contextlib import closing, contextmanager
from typing import TYPE_CHECKING, TypeVar
import pytest
from trio._tests.pytest_plugin import RUN_SLOW
if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence
slow = pytest.mark.skipif(not RUN_SLOW, reason='use --run-slow to run slow tests')
T = TypeVar('T')
buggy_pypy_asyncgens = not TYPE_CHECKING and sys.implementation.name == 'pypy' and (sys.pypy_version_info < (7, 3))
try:
    s = stdlib_socket.socket(stdlib_socket.AF_INET6, stdlib_socket.SOCK_STREAM, 0)
except OSError:
    can_create_ipv6 = False
    can_bind_ipv6 = False
else:
    can_create_ipv6 = True
    with s:
        try:
            s.bind(('::1', 0))
        except OSError:
            can_bind_ipv6 = False
        else:
            can_bind_ipv6 = True
creates_ipv6 = pytest.mark.skipif(not can_create_ipv6, reason='need IPv6')
binds_ipv6 = pytest.mark.skipif(not can_bind_ipv6, reason='need IPv6')

def gc_collect_harder() -> None:
    if False:
        return 10
    for _ in range(5):
        gc.collect()

@contextmanager
def ignore_coroutine_never_awaited_warnings() -> Generator[None, None, None]:
    if False:
        for i in range(10):
            print('nop')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="coroutine '.*' was never awaited")
        try:
            yield
        finally:
            gc_collect_harder()

def _noop(*args: object, **kwargs: object) -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

@contextmanager
def restore_unraisablehook() -> Generator[None, None, None]:
    if False:
        return 10
    (sys.unraisablehook, prev) = (sys.__unraisablehook__, sys.unraisablehook)
    try:
        yield
    finally:
        sys.unraisablehook = prev

def check_sequence_matches(seq: Sequence[T], template: Iterable[T | set[T]]) -> None:
    if False:
        while True:
            i = 10
    i = 0
    for pattern in template:
        if not isinstance(pattern, set):
            pattern = {pattern}
        got = set(seq[i:i + len(pattern)])
        assert got == pattern
        i += len(got)
skip_if_fbsd_pipes_broken = pytest.mark.skipif(sys.platform != 'win32' and hasattr(os, 'uname') and (os.uname().sysname == 'FreeBSD') and (os.uname().release[:4] < '12.2'), reason='hangs on FreeBSD 12.1 and earlier, due to FreeBSD bug #246350')

def create_asyncio_future_in_new_loop() -> asyncio.Future[object]:
    if False:
        return 10
    with closing(asyncio.new_event_loop()) as loop:
        return loop.create_future()