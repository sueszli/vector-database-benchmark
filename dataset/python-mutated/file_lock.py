from __future__ import annotations
import contextlib
import errno
import sys
from collections.abc import Generator
from typing import Callable
if sys.platform == 'win32':
    import msvcrt
    _region = 65535

    @contextlib.contextmanager
    def _locked(fileno: int, blocked_cb: Callable[[], None]) -> Generator[None, None, None]:
        if False:
            for i in range(10):
                print('nop')
        try:
            msvcrt.locking(fileno, msvcrt.LK_NBLCK, _region)
        except OSError:
            blocked_cb()
            while True:
                try:
                    msvcrt.locking(fileno, msvcrt.LK_LOCK, _region)
                except OSError as e:
                    if e.errno != errno.EDEADLOCK:
                        raise
                else:
                    break
        try:
            yield
        finally:
            msvcrt.locking(fileno, msvcrt.LK_UNLCK, _region)
else:
    import fcntl

    @contextlib.contextmanager
    def _locked(fileno: int, blocked_cb: Callable[[], None]) -> Generator[None, None, None]:
        if False:
            for i in range(10):
                print('nop')
        try:
            fcntl.flock(fileno, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            blocked_cb()
            fcntl.flock(fileno, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fileno, fcntl.LOCK_UN)

@contextlib.contextmanager
def lock(path: str, blocked_cb: Callable[[], None]) -> Generator[None, None, None]:
    if False:
        return 10
    with open(path, 'a+') as f:
        with _locked(f.fileno(), blocked_cb):
            yield