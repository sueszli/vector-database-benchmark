"""Pseudo terminal utilities."""
from __future__ import annotations
from select import select
import os
import sys
import tty
from os import close, waitpid
from tty import setraw, tcgetattr, tcsetattr
__all__ = ['openpty', 'fork', 'spawn']
STDIN_FILENO = 0
STDOUT_FILENO = 1
STDERR_FILENO = 2
CHILD = 0

def openpty():
    if False:
        while True:
            i = 10
    'openpty() -> (master_fd, slave_fd)\n    Open a pty master/slave pair, using os.openpty() if possible.'
    try:
        return os.openpty()
    except (AttributeError, OSError):
        pass
    (master_fd, slave_name) = _open_terminal()
    slave_fd = slave_open(slave_name)
    return (master_fd, slave_fd)

def master_open():
    if False:
        return 10
    'master_open() -> (master_fd, slave_name)\n    Open a pty master and return the fd, and the filename of the slave end.\n    Deprecated, use openpty() instead.'
    try:
        (master_fd, slave_fd) = os.openpty()
    except (AttributeError, OSError):
        pass
    else:
        slave_name = os.ttyname(slave_fd)
        os.close(slave_fd)
        return (master_fd, slave_name)
    return _open_terminal()

def _open_terminal():
    if False:
        for i in range(10):
            print('nop')
    'Open pty master and return (master_fd, tty_name).'
    for x in 'pqrstuvwxyzPQRST':
        for y in '0123456789abcdef':
            pty_name = '/dev/pty' + x + y
            try:
                fd = os.open(pty_name, os.O_RDWR)
            except OSError:
                continue
            return (fd, '/dev/tty' + x + y)
    raise OSError('out of pty devices')

def slave_open(tty_name):
    if False:
        while True:
            i = 10
    'slave_open(tty_name) -> slave_fd\n    Open the pty slave and acquire the controlling terminal, returning\n    opened filedescriptor.\n    Deprecated, use openpty() instead.'
    result = os.open(tty_name, os.O_RDWR)
    try:
        from fcntl import ioctl, I_PUSH
    except ImportError:
        return result
    try:
        ioctl(result, I_PUSH, 'ptem')
        ioctl(result, I_PUSH, 'ldterm')
    except OSError:
        pass
    return result

def fork():
    if False:
        return 10
    'fork() -> (pid, master_fd)\n    Fork and make the child a session leader with a controlling terminal.'
    try:
        (pid, fd) = os.forkpty()
    except (AttributeError, OSError):
        pass
    else:
        if pid == CHILD:
            try:
                os.setsid()
            except OSError:
                pass
        return (pid, fd)
    (master_fd, slave_fd) = openpty()
    pid = os.fork()
    if pid == CHILD:
        os.setsid()
        os.close(master_fd)
        os.dup2(slave_fd, STDIN_FILENO)
        os.dup2(slave_fd, STDOUT_FILENO)
        os.dup2(slave_fd, STDERR_FILENO)
        if slave_fd > STDERR_FILENO:
            os.close(slave_fd)
        tmp_fd = os.open(os.ttyname(STDOUT_FILENO), os.O_RDWR)
        os.close(tmp_fd)
    else:
        os.close(slave_fd)
    return (pid, master_fd)

def _writen(fd, data):
    if False:
        for i in range(10):
            print('nop')
    'Write all the data to a descriptor.'
    while data:
        n = os.write(fd, data)
        data = data[n:]

def _read(fd):
    if False:
        return 10
    'Default read function.'
    return os.read(fd, 1024)

def _copy(master_fd, master_read=_read, stdin_read=_read):
    if False:
        i = 10
        return i + 15
    'Parent copy loop.\n    Copies\n            pty master -> standard output   (master_read)\n            standard input -> pty master    (stdin_read)'
    fds = [master_fd, STDIN_FILENO]
    while fds:
        (rfds, _wfds, _xfds) = select(fds, [], [])
        if master_fd in rfds:
            try:
                data = master_read(master_fd)
            except OSError:
                data = b''
            if not data:
                return
            else:
                os.write(STDOUT_FILENO, data)
        if STDIN_FILENO in rfds:
            data = stdin_read(STDIN_FILENO)
            if not data:
                fds.remove(STDIN_FILENO)
            else:
                _writen(master_fd, data)

def spawn(argv, master_read=_read, stdin_read=_read):
    if False:
        for i in range(10):
            print('nop')
    'Create a spawned process.'
    if isinstance(argv, str):
        argv = (argv,)
    sys.audit('pty.spawn', argv)
    (pid, master_fd) = fork()
    if pid == CHILD:
        os.execlp(argv[0], *argv)
    try:
        mode = tcgetattr(STDIN_FILENO)
        setraw(STDIN_FILENO)
        restore = True
    except tty.error:
        restore = False
    try:
        _copy(master_fd, master_read, stdin_read)
    finally:
        if restore:
            tcsetattr(STDIN_FILENO, tty.TCSAFLUSH, mode)
    close(master_fd)
    return waitpid(pid, 0)[1]