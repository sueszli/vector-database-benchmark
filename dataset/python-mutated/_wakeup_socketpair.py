from __future__ import annotations
import contextlib
import signal
import socket
import warnings
from .. import _core
from .._util import is_main_thread

class WakeupSocketpair:

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (self.wakeup_sock, self.write_sock) = socket.socketpair()
        self.wakeup_sock.setblocking(False)
        self.write_sock.setblocking(False)
        self.wakeup_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
        self.write_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1)
        with contextlib.suppress(OSError):
            self.write_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.old_wakeup_fd: int | None = None

    def wakeup_thread_and_signal_safe(self) -> None:
        if False:
            i = 10
            return i + 15
        with contextlib.suppress(BlockingIOError):
            self.write_sock.send(b'\x00')

    async def wait_woken(self) -> None:
        await _core.wait_readable(self.wakeup_sock)
        self.drain()

    def drain(self) -> None:
        if False:
            print('Hello World!')
        try:
            while True:
                self.wakeup_sock.recv(2 ** 16)
        except BlockingIOError:
            pass

    def wakeup_on_signals(self) -> None:
        if False:
            print('Hello World!')
        assert self.old_wakeup_fd is None
        if not is_main_thread():
            return
        fd = self.write_sock.fileno()
        self.old_wakeup_fd = signal.set_wakeup_fd(fd, warn_on_full_buffer=False)
        if self.old_wakeup_fd != -1:
            warnings.warn(RuntimeWarning("It looks like Trio's signal handling code might have collided with another library you're using. If you're running Trio in guest mode, then this might mean you should set host_uses_signal_set_wakeup_fd=True. Otherwise, file a bug on Trio and we'll help you figure out what's going on."), stacklevel=1)

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        self.wakeup_sock.close()
        self.write_sock.close()
        if self.old_wakeup_fd is not None:
            signal.set_wakeup_fd(self.old_wakeup_fd)