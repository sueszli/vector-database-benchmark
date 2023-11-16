from __future__ import annotations
import concurrent.futures
import contextlib
import math
import multiprocessing
import os
import subprocess
import sys
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import MutableMapping
from collections.abc import Sequence
from typing import Any
from typing import Callable
from typing import TypeVar
from pre_commit import parse_shebang
from pre_commit.util import cmd_output_b
from pre_commit.util import cmd_output_p
TArg = TypeVar('TArg')
TRet = TypeVar('TRet')

def cpu_count() -> int:
    if False:
        while True:
            i = 10
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 1

def _environ_size(_env: MutableMapping[str, str] | None=None) -> int:
    if False:
        while True:
            i = 10
    environ = _env if _env is not None else getattr(os, 'environb', os.environ)
    size = 8 * len(environ)
    for (k, v) in environ.items():
        size += len(k) + len(v) + 2
    return size

def _get_platform_max_length() -> int:
    if False:
        while True:
            i = 10
    if os.name == 'posix':
        maximum = os.sysconf('SC_ARG_MAX') - 2048 - _environ_size()
        maximum = max(min(maximum, 2 ** 17), 2 ** 12)
        return maximum
    elif os.name == 'nt':
        return 2 ** 15 - 2048
    else:
        return 2 ** 12

def _command_length(*cmd: str) -> int:
    if False:
        return 10
    full_cmd = ' '.join(cmd)
    if sys.platform == 'win32':
        return len(full_cmd.encode('utf-16le')) // 2
    else:
        return len(full_cmd.encode(sys.getfilesystemencoding()))

class ArgumentTooLongError(RuntimeError):
    pass

def partition(cmd: Sequence[str], varargs: Sequence[str], target_concurrency: int, _max_length: int | None=None) -> tuple[tuple[str, ...], ...]:
    if False:
        print('Hello World!')
    _max_length = _max_length or _get_platform_max_length()
    max_args = max(4, math.ceil(len(varargs) / target_concurrency))
    cmd = tuple(cmd)
    ret = []
    ret_cmd: list[str] = []
    varargs = list(reversed(varargs))
    total_length = _command_length(*cmd) + 1
    while varargs:
        arg = varargs.pop()
        arg_length = _command_length(arg) + 1
        if total_length + arg_length <= _max_length and len(ret_cmd) < max_args:
            ret_cmd.append(arg)
            total_length += arg_length
        elif not ret_cmd:
            raise ArgumentTooLongError(arg)
        else:
            ret.append(cmd + tuple(ret_cmd))
            ret_cmd = []
            total_length = _command_length(*cmd) + 1
            varargs.append(arg)
    ret.append(cmd + tuple(ret_cmd))
    return tuple(ret)

@contextlib.contextmanager
def _thread_mapper(maxsize: int) -> Generator[Callable[[Callable[[TArg], TRet], Iterable[TArg]], Iterable[TRet]], None, None]:
    if False:
        for i in range(10):
            print('nop')
    if maxsize == 1:
        yield map
    else:
        with concurrent.futures.ThreadPoolExecutor(maxsize) as ex:
            yield ex.map

def xargs(cmd: tuple[str, ...], varargs: Sequence[str], *, color: bool=False, target_concurrency: int=1, _max_length: int=_get_platform_max_length(), **kwargs: Any) -> tuple[int, bytes]:
    if False:
        i = 10
        return i + 15
    'A simplified implementation of xargs.\n\n    color: Make a pty if on a platform that supports it\n    target_concurrency: Target number of partitions to run concurrently\n    '
    cmd_fn = cmd_output_p if color else cmd_output_b
    retcode = 0
    stdout = b''
    try:
        cmd = parse_shebang.normalize_cmd(cmd)
    except parse_shebang.ExecutableNotFoundError as e:
        return e.to_output()[:2]
    if sys.platform == 'win32' and cmd[0].lower().endswith(('.bat', '.cmd')):
        cmd_exe = parse_shebang.find_executable('cmd.exe')
        _max_length = 8192 - len(cmd_exe) - len(' /c ') - 1024
    partitions = partition(cmd, varargs, target_concurrency, _max_length)

    def run_cmd_partition(run_cmd: tuple[str, ...]) -> tuple[int, bytes, bytes | None]:
        if False:
            i = 10
            return i + 15
        return cmd_fn(*run_cmd, check=False, stderr=subprocess.STDOUT, **kwargs)
    threads = min(len(partitions), target_concurrency)
    with _thread_mapper(threads) as thread_map:
        results = thread_map(run_cmd_partition, partitions)
        for (proc_retcode, proc_out, _) in results:
            if abs(proc_retcode) > abs(retcode):
                retcode = proc_retcode
            stdout += proc_out
    return (retcode, stdout)