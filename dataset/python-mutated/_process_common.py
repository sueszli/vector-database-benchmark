"""Common utilities for the various process_* implementations.

This file is only meant to be imported by the platform-specific implementations
of subprocess utilities, and it contains tools that are common to all of them.
"""
import subprocess
import shlex
import sys
import os
from IPython.utils import py3compat

def read_no_interrupt(p):
    if False:
        return 10
    'Read from a pipe ignoring EINTR errors.\n\n    This is necessary because when reading from pipes with GUI event loops\n    running in the background, often interrupts are raised that stop the\n    command from completing.'
    import errno
    try:
        return p.read()
    except IOError as err:
        if err.errno != errno.EINTR:
            raise

def process_handler(cmd, callback, stderr=subprocess.PIPE):
    if False:
        i = 10
        return i + 15
    "Open a command in a shell subprocess and execute a callback.\n\n    This function provides common scaffolding for creating subprocess.Popen()\n    calls.  It creates a Popen object and then calls the callback with it.\n\n    Parameters\n    ----------\n    cmd : str or list\n        A command to be executed by the system, using :class:`subprocess.Popen`.\n        If a string is passed, it will be run in the system shell. If a list is\n        passed, it will be used directly as arguments.\n    callback : callable\n        A one-argument function that will be called with the Popen object.\n    stderr : file descriptor number, optional\n        By default this is set to ``subprocess.PIPE``, but you can also pass the\n        value ``subprocess.STDOUT`` to force the subprocess' stderr to go into\n        the same file descriptor as its stdout.  This is useful to read stdout\n        and stderr combined in the order they are generated.\n\n    Returns\n    -------\n    The return value of the provided callback is returned.\n    "
    sys.stdout.flush()
    sys.stderr.flush()
    close_fds = sys.platform != 'win32'
    shell = isinstance(cmd, str)
    executable = None
    if shell and os.name == 'posix' and ('SHELL' in os.environ):
        executable = os.environ['SHELL']
    p = subprocess.Popen(cmd, shell=shell, executable=executable, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, close_fds=close_fds)
    try:
        out = callback(p)
    except KeyboardInterrupt:
        print('^C')
        sys.stdout.flush()
        sys.stderr.flush()
        out = None
    finally:
        if p.returncode is None:
            try:
                p.terminate()
                p.poll()
            except OSError:
                pass
        if p.returncode is None:
            try:
                p.kill()
            except OSError:
                pass
    return out

def getoutput(cmd):
    if False:
        while True:
            i = 10
    'Run a command and return its stdout/stderr as a string.\n\n    Parameters\n    ----------\n    cmd : str or list\n        A command to be executed in the system shell.\n\n    Returns\n    -------\n    output : str\n        A string containing the combination of stdout and stderr from the\n    subprocess, in whatever order the subprocess originally wrote to its\n    file descriptors (so the order of the information in this string is the\n    correct order as would be seen if running the command in a terminal).\n    '
    out = process_handler(cmd, lambda p: p.communicate()[0], subprocess.STDOUT)
    if out is None:
        return ''
    return py3compat.decode(out)

def getoutputerror(cmd):
    if False:
        return 10
    'Return (standard output, standard error) of executing cmd in a shell.\n\n    Accepts the same arguments as os.system().\n\n    Parameters\n    ----------\n    cmd : str or list\n        A command to be executed in the system shell.\n\n    Returns\n    -------\n    stdout : str\n    stderr : str\n    '
    return get_output_error_code(cmd)[:2]

def get_output_error_code(cmd):
    if False:
        return 10
    'Return (standard output, standard error, return code) of executing cmd\n    in a shell.\n\n    Accepts the same arguments as os.system().\n\n    Parameters\n    ----------\n    cmd : str or list\n        A command to be executed in the system shell.\n\n    Returns\n    -------\n    stdout : str\n    stderr : str\n    returncode: int\n    '
    (out_err, p) = process_handler(cmd, lambda p: (p.communicate(), p))
    if out_err is None:
        return ('', '', p.returncode)
    (out, err) = out_err
    return (py3compat.decode(out), py3compat.decode(err), p.returncode)

def arg_split(s, posix=False, strict=True):
    if False:
        return 10
    "Split a command line's arguments in a shell-like manner.\n\n    This is a modified version of the standard library's shlex.split()\n    function, but with a default of posix=False for splitting, so that quotes\n    in inputs are respected.\n\n    if strict=False, then any errors shlex.split would raise will result in the\n    unparsed remainder being the last element of the list, rather than raising.\n    This is because we sometimes use arg_split to parse things other than\n    command-line args.\n    "
    lex = shlex.shlex(s, posix=posix)
    lex.whitespace_split = True
    lex.commenters = ''
    tokens = []
    while True:
        try:
            tokens.append(next(lex))
        except StopIteration:
            break
        except ValueError:
            if strict:
                raise
            tokens.append(lex.token)
            break
    return tokens