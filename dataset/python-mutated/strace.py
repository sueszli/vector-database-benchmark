"""Support for running strace against the current process."""
from __future__ import absolute_import
import os
import signal
import subprocess
import tempfile
from bzrlib import errors

def strace(function, *args, **kwargs):
    if False:
        print('Hello World!')
    'Invoke strace on function.\n\n    :return: a tuple: function-result, a StraceResult.\n    '
    return strace_detailed(function, args, kwargs)

def strace_detailed(function, args, kwargs, follow_children=True):
    if False:
        return 10
    log_file = tempfile.NamedTemporaryFile()
    log_file_fd = log_file.fileno()
    err_file = tempfile.NamedTemporaryFile()
    pid = os.getpid()
    strace_cmd = ['strace', '-r', '-tt', '-p', str(pid), '-o', log_file.name]
    if follow_children:
        strace_cmd.append('-f')
    proc = subprocess.Popen(strace_cmd, stdout=subprocess.PIPE, stderr=err_file.fileno())
    attached_notice = proc.stdout.readline()
    result = function(*args, **kwargs)
    os.kill(proc.pid, signal.SIGQUIT)
    proc.communicate()
    log_file.seek(0)
    log = log_file.read()
    log_file.close()
    err_file.seek(0)
    err_messages = err_file.read()
    err_file.close()
    if err_messages.startswith('attach: ptrace(PTRACE_ATTACH,'):
        raise StraceError(err_messages=err_messages)
    return (result, StraceResult(log, err_messages))

class StraceError(errors.BzrError):
    _fmt = 'strace failed: %(err_messages)s'

class StraceResult(object):
    """The result of stracing a function."""

    def __init__(self, raw_log, err_messages):
        if False:
            print('Hello World!')
        'Create a StraceResult.\n\n        :param raw_log: The output that strace created.\n        '
        self.raw_log = raw_log
        self.err_messages = err_messages