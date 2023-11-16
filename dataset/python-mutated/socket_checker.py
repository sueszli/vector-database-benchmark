"""Select / poll helper"""
from __future__ import annotations
import errno
import select
import sys
from typing import Any, Optional, cast
_HAVE_POLL = hasattr(select, 'poll') and (not sys.platform.startswith('java'))
_SelectError = getattr(select, 'error', OSError)

def _errno_from_exception(exc: BaseException) -> Optional[int]:
    if False:
        for i in range(10):
            print('nop')
    if hasattr(exc, 'errno'):
        return cast(int, exc.errno)
    if exc.args:
        return cast(int, exc.args[0])
    return None

class SocketChecker:

    def __init__(self) -> None:
        if False:
            return 10
        self._poller: Optional[select.poll]
        if _HAVE_POLL:
            self._poller = select.poll()
        else:
            self._poller = None

    def select(self, sock: Any, read: bool=False, write: bool=False, timeout: Optional[float]=0) -> bool:
        if False:
            print('Hello World!')
        'Select for reads or writes with a timeout in seconds (or None).\n\n        Returns True if the socket is readable/writable, False on timeout.\n        '
        res: Any
        while True:
            try:
                if self._poller:
                    mask = select.POLLERR | select.POLLHUP
                    if read:
                        mask = mask | select.POLLIN | select.POLLPRI
                    if write:
                        mask = mask | select.POLLOUT
                    self._poller.register(sock, mask)
                    try:
                        timeout_ = None if timeout is None else timeout * 1000
                        res = self._poller.poll(timeout_)
                        return bool(res)
                    finally:
                        self._poller.unregister(sock)
                else:
                    rlist = [sock] if read else []
                    wlist = [sock] if write else []
                    res = select.select(rlist, wlist, [sock], timeout)
                    return any(res)
            except (_SelectError, OSError) as exc:
                if _errno_from_exception(exc) in (errno.EINTR, errno.EAGAIN):
                    continue
                raise

    def socket_closed(self, sock: Any) -> bool:
        if False:
            return 10
        'Return True if we know socket has been closed, False otherwise.'
        try:
            return self.select(sock, read=True)
        except (RuntimeError, KeyError):
            raise
        except ValueError:
            return True
        except Exception:
            return True