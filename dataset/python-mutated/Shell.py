from contextlib import contextmanager
import platform
import shlex
from subprocess import PIPE, Popen
from shutil import which

class ShellCommandResult(tuple):
    """
    The result of a :func:`coalib.misc.run_shell_command` call.

    It is based on a ``(stdout, stderr)`` string tuple like it is returned
    form ``subprocess.Popen.communicate`` and was originally returned from
    :func:`coalib.misc.run_shell_command`. So it is backwards-compatible.

    It additionally stores the return ``.code``:

    >>> import sys
    >>> process = Popen([sys.executable, '-c',
    ...                  'import sys; print(sys.stdin.readline().strip() +'
    ...                  '                  " processed")'],
    ...                 stdin=PIPE, stdout=PIPE, stderr=PIPE,
    ...                 universal_newlines=True)

    >>> stdout, stderr = process.communicate(input='data')
    >>> stderr
    ''
    >>> result = ShellCommandResult(process.returncode, stdout, stderr)
    >>> result[0]
    'data processed\\n'
    >>> result[1]
    ''
    >>> result.code
    0
    """

    def __new__(cls, code, stdout, stderr):
        if False:
            i = 10
            return i + 15
        '\n        Creates the basic tuple from `stdout` and `stderr`.\n        '
        return tuple.__new__(cls, (stdout, stderr))

    def __init__(self, code, stdout, stderr):
        if False:
            print('Hello World!')
        '\n        Stores the return `code`.\n        '
        self.code = code

@contextmanager
def run_interactive_shell_command(command, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Runs a single command in shell and provides stdout, stderr and stdin\n    streams.\n\n    This function creates a context manager that sets up the process (using\n    ``subprocess.Popen()``), returns to caller and waits for process to exit on\n    leaving.\n\n    By default the process is opened in ``universal_newlines`` mode and creates\n    pipes for all streams (stdout, stderr and stdin) using ``subprocess.PIPE``\n    special value. These pipes are closed automatically, so if you want to get\n    the contents of the streams you should retrieve them before the context\n    manager exits.\n\n    >>> with run_interactive_shell_command(["echo", "TEXT"]) as p:\n    ...     stdout = p.stdout\n    ...     stdout_text = stdout.read()\n    >>> stdout_text\n    \'TEXT\\n\'\n    >>> stdout.closed\n    True\n\n    Custom streams provided are not closed except of ``subprocess.PIPE``.\n\n    >>> from tempfile import TemporaryFile\n    >>> stream = TemporaryFile()\n    >>> with run_interactive_shell_command(["echo", "TEXT"],\n    ...                                    stdout=stream) as p:\n    ...     stderr = p.stderr\n    >>> stderr.closed\n    True\n    >>> stream.closed\n    False\n\n    :param command: The command to run on shell. This parameter can either\n                    be a sequence of arguments that are directly passed to\n                    the process or a string. A string gets splitted beforehand\n                    using ``shlex.split()``. If providing ``shell=True`` as a\n                    keyword-argument, no ``shlex.split()`` is performed and the\n                    command string goes directly to ``subprocess.Popen()``.\n    :param kwargs:  Additional keyword arguments to pass to\n                    ``subprocess.Popen`` that are used to spawn the process.\n    :return:        A context manager yielding the process started from the\n                    command.\n    '
    if not kwargs.get('shell', False) and isinstance(command, str):
        command = shlex.split(command)
    else:
        command = list(command)
    if platform.system() == 'Windows':
        command[0] = which(command[0])
    args = {'stdout': PIPE, 'stderr': PIPE, 'stdin': PIPE, 'universal_newlines': True}
    args.update(kwargs)
    process = Popen(command, **args)
    try:
        yield process
    finally:
        if args['stdout'] is PIPE:
            process.stdout.close()
        if args['stderr'] is PIPE:
            process.stderr.close()
        if args['stdin'] is PIPE:
            process.stdin.close()
        process.wait()

def run_shell_command(command, stdin=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Runs a single command in shell and returns the read stdout and stderr data.\n\n    This function waits for the process (created using ``subprocess.Popen()``)\n    to exit. Effectively it wraps ``run_interactive_shell_command()`` and uses\n    ``communicate()`` on the process.\n\n    See also ``run_interactive_shell_command()``.\n\n    :param command: The command to run on shell. This parameter can either\n                    be a sequence of arguments that are directly passed to\n                    the process or a string. A string gets splitted beforehand\n                    using ``shlex.split()``.\n    :param stdin:   Initial input to send to the process.\n    :param kwargs:  Additional keyword arguments to pass to\n                    ``subprocess.Popen`` that is used to spawn the process.\n    :return:        A tuple with ``(stdoutstring, stderrstring)``.\n    '
    with run_interactive_shell_command(command, **kwargs) as p:
        ret = p.communicate(stdin)
    return ShellCommandResult(p.returncode, *ret)