"""
Define the fork_subprocess.Process class, which uses the
'multiprocessing' library to fork a child process, then expose its
stdout and stderr as 'asyncio.StreamReaders' in the parent.
"""
import asyncio
import multiprocessing
import os
from typing import Callable
DEFAULT_LIMIT = 2 ** 16

async def reader_for_pipe_fd(fd: int, limit: int) -> asyncio.StreamReader:
    """
    Return a StreamReader for pipe file descriptor 'fd'.

    The 'limit' defines the maximum size of a single 'read' operation to
    avoid the possibility of deadlock.
    """
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(loop=loop, limit=limit)
    fd_file = os.fdopen(fd, 'rb', closefd=False)
    await loop.connect_read_pipe(lambda : asyncio.StreamReaderProtocol(reader, loop=loop), fd_file)
    return reader

class Pipe:
    """
    Pair of file descriptors created by 'os.pipe()'.
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        (self.read_fd, self.write_fd) = os.pipe()

    def set_write_inheritable(self) -> None:
        if False:
            while True:
                i = 10
        os.set_inheritable(self.write_fd, True)

    def close_write_end(self) -> None:
        if False:
            while True:
                i = 10
        os.close(self.write_fd)

    def close_read_end(self) -> None:
        if False:
            print('Hello World!')
        os.close(self.read_fd)

class Process:
    """
    This class manages a subprocess created by 'os.fork()', and has an
    interface somewhat similar to asyncio.subprocess.Process,
    particularly exposing stdout and stderr as StreamReaders.
    """

    def __init__(self, func: Callable[[], None], stdout_pipe: Pipe, stderr_pipe: Pipe, stdout: asyncio.StreamReader, stderr: asyncio.StreamReader) -> None:
        if False:
            while True:
                i = 10
        "\n        This constructor is intended to be private to this module.  Use\n        'start_fork_subprocess()' to make an object.\n        "
        self._func = func
        self._stdout_pipe = stdout_pipe
        self._stderr_pipe = stderr_pipe
        self.stdout = stdout
        self.stderr = stderr
        writeFds = (self._stdout_pipe.write_fd, self._stderr_pipe.write_fd)
        mpctx = multiprocessing.get_context('fork')
        multiprocessing.process.current_process().daemon = False
        self._child = mpctx.Process(target=self._callFunc, args=writeFds)
        self._child.start()

    def _callFunc(self, outfd: int, errfd: int) -> None:
        if False:
            print('Hello World!')
        "\n        Invoke '_func(_arg)' after redirecting stdout and stderr.\n        "
        os.dup2(outfd, 1)
        os.close(outfd)
        os.dup2(errfd, 2)
        os.close(errfd)
        self._func()

    def wait(self) -> int:
        if False:
            print('Hello World!')
        '\n        Synchronously wait for the child process to terminate, and\n        return its exit code.\n        '
        self._child.join()
        self._stdout_pipe.close_read_end()
        self._stderr_pipe.close_read_end()
        assert self._child.exitcode is not None
        return self._child.exitcode

async def start_fork_subprocess(func: Callable[[], None], limit: int=DEFAULT_LIMIT) -> Process:
    """
    Fork a child process that executes 'func()'.

    Return the Process object that manages the child.
    """
    stdout_pipe = Pipe()
    stderr_pipe = Pipe()
    stdout_pipe.set_write_inheritable()
    stderr_pipe.set_write_inheritable()
    stdout = await reader_for_pipe_fd(stdout_pipe.read_fd, limit)
    stderr = await reader_for_pipe_fd(stderr_pipe.read_fd, limit)
    ret = Process(func, stdout_pipe, stderr_pipe, stdout, stderr)
    stdout_pipe.close_write_end()
    stderr_pipe.close_write_end()
    return ret