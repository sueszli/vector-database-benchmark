"""
Utility functions for dealing with POSIX file descriptors.
"""
import errno
import os
try:
    import fcntl as _fcntl
except ImportError:
    fcntl = None
else:
    fcntl = _fcntl
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST

def setNonBlocking(fd):
    if False:
        return 10
    '\n    Set the file description of the given file descriptor to non-blocking.\n    '
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    flags = flags | os.O_NONBLOCK
    fcntl.fcntl(fd, fcntl.F_SETFL, flags)

def setBlocking(fd):
    if False:
        i = 10
        return i + 15
    '\n    Set the file description of the given file descriptor to blocking.\n    '
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    flags = flags & ~os.O_NONBLOCK
    fcntl.fcntl(fd, fcntl.F_SETFL, flags)
if fcntl is None:
    _setCloseOnExec = _unsetCloseOnExec = lambda fd: None
else:

    def _setCloseOnExec(fd):
        if False:
            i = 10
            return i + 15
        '\n        Make a file descriptor close-on-exec.\n        '
        flags = fcntl.fcntl(fd, fcntl.F_GETFD)
        flags = flags | fcntl.FD_CLOEXEC
        fcntl.fcntl(fd, fcntl.F_SETFD, flags)

    def _unsetCloseOnExec(fd):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make a file descriptor close-on-exec.\n        '
        flags = fcntl.fcntl(fd, fcntl.F_GETFD)
        flags = flags & ~fcntl.FD_CLOEXEC
        fcntl.fcntl(fd, fcntl.F_SETFD, flags)

def readFromFD(fd, callback):
    if False:
        return 10
    "\n    Read from file descriptor, calling callback with resulting data.\n\n    If successful, call 'callback' with a single argument: the\n    resulting data.\n\n    Returns same thing FileDescriptor.doRead would: CONNECTION_LOST,\n    CONNECTION_DONE, or None.\n\n    @type fd: C{int}\n    @param fd: non-blocking file descriptor to be read from.\n    @param callback: a callable which accepts a single argument. If\n    data is read from the file descriptor it will be called with this\n    data. Handling exceptions from calling the callback is up to the\n    caller.\n\n    Note that if the descriptor is still connected but no data is read,\n    None will be returned but callback will not be called.\n\n    @return: CONNECTION_LOST on error, CONNECTION_DONE when fd is\n    closed, otherwise None.\n    "
    try:
        output = os.read(fd, 8192)
    except OSError as ioe:
        if ioe.args[0] in (errno.EAGAIN, errno.EINTR):
            return
        else:
            return CONNECTION_LOST
    if not output:
        return CONNECTION_DONE
    callback(output)

def writeToFD(fd, data):
    if False:
        return 10
    '\n    Write data to file descriptor.\n\n    Returns same thing FileDescriptor.writeSomeData would.\n\n    @type fd: C{int}\n    @param fd: non-blocking file descriptor to be written to.\n    @type data: C{str} or C{buffer}\n    @param data: bytes to write to fd.\n\n    @return: number of bytes written, or CONNECTION_LOST.\n    '
    try:
        return os.write(fd, data)
    except OSError as io:
        if io.errno in (errno.EAGAIN, errno.EINTR):
            return 0
        return CONNECTION_LOST
__all__ = ['setNonBlocking', 'setBlocking', 'readFromFD', 'writeToFD']