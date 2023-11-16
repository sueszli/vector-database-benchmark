"""Posix-specific implementation of process utilities.

This file is only meant to be imported by process.py, not by end-users.
"""
import errno
import os
import subprocess as sp
import sys
import pexpect
from ._process_common import getoutput, arg_split
from IPython.utils.encoding import DEFAULT_ENCODING

class ProcessHandler(object):
    """Execute subprocesses under the control of pexpect.
    """
    read_timeout = 0.05
    terminate_timeout = 0.2
    logfile = None
    _sh = None

    @property
    def sh(self):
        if False:
            i = 10
            return i + 15
        if self._sh is None:
            shell_name = os.environ.get('SHELL', 'sh')
            self._sh = pexpect.which(shell_name)
            if self._sh is None:
                raise OSError('"{}" shell not found'.format(shell_name))
        return self._sh

    def __init__(self, logfile=None, read_timeout=None, terminate_timeout=None):
        if False:
            while True:
                i = 10
        'Arguments are used for pexpect calls.'
        self.read_timeout = ProcessHandler.read_timeout if read_timeout is None else read_timeout
        self.terminate_timeout = ProcessHandler.terminate_timeout if terminate_timeout is None else terminate_timeout
        self.logfile = sys.stdout if logfile is None else logfile

    def getoutput(self, cmd):
        if False:
            while True:
                i = 10
        'Run a command and return its stdout/stderr as a string.\n\n        Parameters\n        ----------\n        cmd : str\n            A command to be executed in the system shell.\n\n        Returns\n        -------\n        output : str\n            A string containing the combination of stdout and stderr from the\n        subprocess, in whatever order the subprocess originally wrote to its\n        file descriptors (so the order of the information in this string is the\n        correct order as would be seen if running the command in a terminal).\n        '
        try:
            return pexpect.run(self.sh, args=['-c', cmd]).replace('\r\n', '\n')
        except KeyboardInterrupt:
            print('^C', file=sys.stderr, end='')

    def getoutput_pexpect(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        'Run a command and return its stdout/stderr as a string.\n\n        Parameters\n        ----------\n        cmd : str\n            A command to be executed in the system shell.\n\n        Returns\n        -------\n        output : str\n            A string containing the combination of stdout and stderr from the\n        subprocess, in whatever order the subprocess originally wrote to its\n        file descriptors (so the order of the information in this string is the\n        correct order as would be seen if running the command in a terminal).\n        '
        try:
            return pexpect.run(self.sh, args=['-c', cmd]).replace('\r\n', '\n')
        except KeyboardInterrupt:
            print('^C', file=sys.stderr, end='')

    def system(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        "Execute a command in a subshell.\n\n        Parameters\n        ----------\n        cmd : str\n            A command to be executed in the system shell.\n\n        Returns\n        -------\n        int : child's exitstatus\n        "
        enc = DEFAULT_ENCODING
        patterns = [pexpect.TIMEOUT, pexpect.EOF]
        EOF_index = patterns.index(pexpect.EOF)
        out_size = 0
        try:
            if hasattr(pexpect, 'spawnb'):
                child = pexpect.spawnb(self.sh, args=['-c', cmd])
            else:
                child = pexpect.spawn(self.sh, args=['-c', cmd])
            flush = sys.stdout.flush
            while True:
                res_idx = child.expect_list(patterns, self.read_timeout)
                print(child.before[out_size:].decode(enc, 'replace'), end='')
                flush()
                if res_idx == EOF_index:
                    break
                out_size = len(child.before)
        except KeyboardInterrupt:
            child.sendline(chr(3))
            try:
                out_size = len(child.before)
                child.expect_list(patterns, self.terminate_timeout)
                print(child.before[out_size:].decode(enc, 'replace'), end='')
                sys.stdout.flush()
            except KeyboardInterrupt:
                pass
            finally:
                child.terminate(force=True)
        child.isalive()
        if child.exitstatus is None:
            if child.signalstatus is None:
                return 0
            return -child.signalstatus
        if child.exitstatus > 128:
            return -(child.exitstatus - 128)
        return child.exitstatus
system = ProcessHandler().system

def check_pid(pid):
    if False:
        for i in range(10):
            print('nop')
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            return False
        elif err.errno == errno.EPERM:
            return True
        raise
    else:
        return True