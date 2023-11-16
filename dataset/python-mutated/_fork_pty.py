"""Substitute for the forkpty system call, to support Solaris.
"""
import os
import errno
from pty import STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO, CHILD
from .util import PtyProcessError

def fork_pty():
    if False:
        while True:
            i = 10
    "This implements a substitute for the forkpty system call. This\n    should be more portable than the pty.fork() function. Specifically,\n    this should work on Solaris.\n\n    Modified 10.06.05 by Geoff Marshall: Implemented __fork_pty() method to\n    resolve the issue with Python's pty.fork() not supporting Solaris,\n    particularly ssh. Based on patch to posixmodule.c authored by Noah\n    Spurrier::\n\n        http://mail.python.org/pipermail/python-dev/2003-May/035281.html\n\n    "
    (parent_fd, child_fd) = os.openpty()
    if parent_fd < 0 or child_fd < 0:
        raise OSError('os.openpty() failed')
    pid = os.fork()
    if pid == CHILD:
        os.close(parent_fd)
        pty_make_controlling_tty(child_fd)
        os.dup2(child_fd, STDIN_FILENO)
        os.dup2(child_fd, STDOUT_FILENO)
        os.dup2(child_fd, STDERR_FILENO)
    else:
        os.close(child_fd)
    return (pid, parent_fd)

def pty_make_controlling_tty(tty_fd):
    if False:
        i = 10
        return i + 15
    'This makes the pseudo-terminal the controlling tty. This should be\n    more portable than the pty.fork() function. Specifically, this should\n    work on Solaris. '
    child_name = os.ttyname(tty_fd)
    try:
        fd = os.open('/dev/tty', os.O_RDWR | os.O_NOCTTY)
        os.close(fd)
    except OSError as err:
        if err.errno != errno.ENXIO:
            raise
    os.setsid()
    try:
        fd = os.open('/dev/tty', os.O_RDWR | os.O_NOCTTY)
        os.close(fd)
        raise PtyProcessError('OSError of errno.ENXIO should be raised.')
    except OSError as err:
        if err.errno != errno.ENXIO:
            raise
    fd = os.open(child_name, os.O_RDWR)
    os.close(fd)
    fd = os.open('/dev/tty', os.O_WRONLY)
    os.close(fd)