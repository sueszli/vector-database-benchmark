import os.path
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '\\..'))
import trio
print(trio.__file__)
import socket
import trio.testing
from trio._core._io_windows import _afd_helper_handle, _check, _get_base_socket
from trio._core._windows_cffi import AFDPollFlags, ErrorCodes, IoControlCodes, ffi, kernel32

class AFDLab:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._afd = _afd_helper_handle()
        trio.lowlevel.register_with_iocp(self._afd)

    async def afd_poll(self, sock, flags, *, exclusive=0):
        print(f'Starting a poll for {flags!r}')
        lpOverlapped = ffi.new('LPOVERLAPPED')
        poll_info = ffi.new('AFD_POLL_INFO *')
        poll_info.Timeout = 2 ** 63 - 1
        poll_info.NumberOfHandles = 1
        poll_info.Exclusive = exclusive
        poll_info.Handles[0].Handle = _get_base_socket(sock)
        poll_info.Handles[0].Status = 0
        poll_info.Handles[0].Events = flags
        try:
            _check(kernel32.DeviceIoControl(self._afd, IoControlCodes.IOCTL_AFD_POLL, poll_info, ffi.sizeof('AFD_POLL_INFO'), poll_info, ffi.sizeof('AFD_POLL_INFO'), ffi.NULL, lpOverlapped))
        except OSError as exc:
            if exc.winerror != ErrorCodes.ERROR_IO_PENDING:
                raise
        try:
            await trio.lowlevel.wait_overlapped(self._afd, lpOverlapped)
        except:
            print(f'Poll for {flags!r}: {sys.exc_info()[1]!r}')
            raise
        out_flags = AFDPollFlags(poll_info.Handles[0].Events)
        print(f'Poll for {flags!r}: got {out_flags!r}')
        return out_flags

def fill_socket(sock):
    if False:
        print('Hello World!')
    try:
        while True:
            sock.send(b'x' * 65536)
    except BlockingIOError:
        pass

async def main():
    afdlab = AFDLab()
    (a, b) = socket.socketpair()
    a.setblocking(False)
    b.setblocking(False)
    fill_socket(a)
    while True:
        print('-- Iteration start --')
        async with trio.open_nursery() as nursery:
            nursery.start_soon(afdlab.afd_poll, a, AFDPollFlags.AFD_POLL_SEND)
            await trio.sleep(2)
            nursery.start_soon(afdlab.afd_poll, a, AFDPollFlags.AFD_POLL_RECEIVE)
            await trio.sleep(2)
            print('Sending another byte')
            b.send(b'x')
            await trio.sleep(2)
            nursery.cancel_scope.cancel()
trio.run(main)