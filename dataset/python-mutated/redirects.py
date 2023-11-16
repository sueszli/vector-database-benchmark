import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from functools import partial
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
logger = logging.getLogger(__name__)

def get_libc():
    if False:
        for i in range(10):
            print('nop')
    if IS_WINDOWS or IS_MACOS:
        logger.warning('NOTE: Redirects are currently not supported in Windows or MacOs.')
        return None
    else:
        return ctypes.CDLL('libc.so.6')
libc = get_libc()

def _c_std(stream: str):
    if False:
        while True:
            i = 10
    return ctypes.c_void_p.in_dll(libc, stream)

def _python_std(stream: str):
    if False:
        return 10
    return {'stdout': sys.stdout, 'stderr': sys.stderr}[stream]
_VALID_STD = {'stdout', 'stderr'}

@contextmanager
def redirect(std: str, to_file: str):
    if False:
        while True:
            i = 10
    '\n    Redirect ``std`` (one of ``"stdout"`` or ``"stderr"``) to a file in the path specified by ``to_file``.\n\n    This method redirects the underlying std file descriptor (not just python\'s ``sys.stdout|stderr``).\n    See usage for details.\n\n    Directory of ``dst_filename`` is assumed to exist and the destination file\n    is overwritten if it already exists.\n\n    .. note:: Due to buffering cross source writes are not guaranteed to\n              appear in wall-clock order. For instance in the example below\n              it is possible for the C-outputs to appear before the python\n              outputs in the log file.\n\n    Usage:\n\n    ::\n\n     # syntactic-sugar for redirect("stdout", "tmp/stdout.log")\n     with redirect_stdout("/tmp/stdout.log"):\n        print("python stdouts are redirected")\n        libc = ctypes.CDLL("libc.so.6")\n        libc.printf(b"c stdouts are also redirected"\n        os.system("echo system stdouts are also redirected")\n\n     print("stdout restored")\n\n    '
    if std not in _VALID_STD:
        raise ValueError(f'unknown standard stream <{std}>, must be one of {_VALID_STD}')
    c_std = _c_std(std)
    python_std = _python_std(std)
    std_fd = python_std.fileno()

    def _redirect(dst):
        if False:
            while True:
                i = 10
        libc.fflush(c_std)
        python_std.flush()
        os.dup2(dst.fileno(), std_fd)
    with os.fdopen(os.dup(std_fd)) as orig_std, open(to_file, mode='w+b') as dst:
        _redirect(dst)
        try:
            yield
        finally:
            _redirect(orig_std)
redirect_stdout = partial(redirect, 'stdout')
redirect_stderr = partial(redirect, 'stderr')