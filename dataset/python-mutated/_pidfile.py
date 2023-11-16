"""
PID file.
"""
from __future__ import annotations
import errno
from os import getpid, kill, name as SYSTEM_NAME
from types import TracebackType
from typing import Any, Optional, Type
from zope.interface import Interface, implementer
from twisted.logger import Logger
from twisted.python.filepath import FilePath

class IPIDFile(Interface):
    """
    Manages a file that remembers a process ID.
    """

    def read() -> int:
        if False:
            for i in range(10):
                print('nop')
        "\n        Read the process ID stored in this PID file.\n\n        @return: The contained process ID.\n\n        @raise NoPIDFound: If this PID file does not exist.\n        @raise EnvironmentError: If this PID file cannot be read.\n        @raise ValueError: If this PID file's content is invalid.\n        "

    def writeRunningPID() -> None:
        if False:
            i = 10
            return i + 15
        '\n        Store the PID of the current process in this PID file.\n\n        @raise EnvironmentError: If this PID file cannot be written.\n        '

    def remove() -> None:
        if False:
            while True:
                i = 10
        '\n        Remove this PID file.\n\n        @raise EnvironmentError: If this PID file cannot be removed.\n        '

    def isRunning() -> bool:
        if False:
            while True:
                i = 10
        "\n        Determine whether there is a running process corresponding to the PID\n        in this PID file.\n\n        @return: True if this PID file contains a PID and a process with that\n            PID is currently running; false otherwise.\n\n        @raise EnvironmentError: If this PID file cannot be read.\n        @raise InvalidPIDFileError: If this PID file's content is invalid.\n        @raise StalePIDFileError: If this PID file's content refers to a PID\n            for which there is no corresponding running process.\n        "

    def __enter__() -> 'IPIDFile':
        if False:
            print('Hello World!')
        '\n        Enter a context using this PIDFile.\n\n        Writes the PID file with the PID of the running process.\n\n        @raise AlreadyRunningError: A process corresponding to the PID in this\n            PID file is already running.\n        '

    def __exit__(excType: Optional[Type[BaseException]], excValue: Optional[BaseException], traceback: Optional[TracebackType]) -> Optional[bool]:
        if False:
            print('Hello World!')
        '\n        Exit a context using this PIDFile.\n\n        Removes the PID file.\n        '

@implementer(IPIDFile)
class PIDFile:
    """
    Concrete implementation of L{IPIDFile}.

    This implementation is presently not supported on non-POSIX platforms.
    Specifically, calling L{PIDFile.isRunning} will raise
    L{NotImplementedError}.
    """
    _log = Logger()

    @staticmethod
    def _format(pid: int) -> bytes:
        if False:
            while True:
                i = 10
        "\n        Format a PID file's content.\n\n        @param pid: A process ID.\n\n        @return: Formatted PID file contents.\n        "
        return f'{int(pid)}\n'.encode()

    def __init__(self, filePath: FilePath[Any]) -> None:
        if False:
            while True:
                i = 10
        '\n        @param filePath: The path to the PID file on disk.\n        '
        self.filePath = filePath

    def read(self) -> int:
        if False:
            print('Hello World!')
        pidString = b''
        try:
            with self.filePath.open() as fh:
                for pidString in fh:
                    break
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise NoPIDFound('PID file does not exist')
            raise
        try:
            return int(pidString)
        except ValueError:
            raise InvalidPIDFileError(f'non-integer PID value in PID file: {pidString!r}')

    def _write(self, pid: int) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Store a PID in this PID file.\n\n        @param pid: A PID to store.\n\n        @raise EnvironmentError: If this PID file cannot be written.\n        '
        self.filePath.setContent(self._format(pid=pid))

    def writeRunningPID(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._write(getpid())

    def remove(self) -> None:
        if False:
            return 10
        self.filePath.remove()

    def isRunning(self) -> bool:
        if False:
            return 10
        try:
            pid = self.read()
        except NoPIDFound:
            return False
        if SYSTEM_NAME == 'posix':
            return self._pidIsRunningPOSIX(pid)
        else:
            raise NotImplementedError(f'isRunning is not implemented on {SYSTEM_NAME}')

    @staticmethod
    def _pidIsRunningPOSIX(pid: int) -> bool:
        if False:
            while True:
                i = 10
        "\n        POSIX implementation for running process check.\n\n        Determine whether there is a running process corresponding to the given\n        PID.\n\n        @param pid: The PID to check.\n\n        @return: True if the given PID is currently running; false otherwise.\n\n        @raise EnvironmentError: If this PID file cannot be read.\n        @raise InvalidPIDFileError: If this PID file's content is invalid.\n        @raise StalePIDFileError: If this PID file's content refers to a PID\n            for which there is no corresponding running process.\n        "
        try:
            kill(pid, 0)
        except OSError as e:
            if e.errno == errno.ESRCH:
                raise StalePIDFileError('PID file refers to non-existing process')
            elif e.errno == errno.EPERM:
                return True
            else:
                raise
        else:
            return True

    def __enter__(self) -> 'PIDFile':
        if False:
            i = 10
            return i + 15
        try:
            if self.isRunning():
                raise AlreadyRunningError()
        except StalePIDFileError:
            self._log.info('Replacing stale PID file: {log_source}')
        self.writeRunningPID()
        return self

    def __exit__(self, excType: Optional[Type[BaseException]], excValue: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        if False:
            i = 10
            return i + 15
        self.remove()
        return None

@implementer(IPIDFile)
class NonePIDFile:
    """
    PID file implementation that does nothing.

    This is meant to be used as a "active None" object in place of a PID file
    when no PID file is desired.
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pass

    def read(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        raise NoPIDFound('PID file does not exist')

    def _write(self, pid: int) -> None:
        if False:
            print('Hello World!')
        '\n        Store a PID in this PID file.\n\n        @param pid: A PID to store.\n\n        @raise EnvironmentError: If this PID file cannot be written.\n\n        @note: This implementation always raises an L{EnvironmentError}.\n        '
        raise OSError(errno.EPERM, 'Operation not permitted')

    def writeRunningPID(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._write(0)

    def remove(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise OSError(errno.ENOENT, 'No such file or directory')

    def isRunning(self) -> bool:
        if False:
            print('Hello World!')
        return False

    def __enter__(self) -> 'NonePIDFile':
        if False:
            print('Hello World!')
        return self

    def __exit__(self, excType: Optional[Type[BaseException]], excValue: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        if False:
            return 10
        return None
nonePIDFile: IPIDFile = NonePIDFile()

class AlreadyRunningError(Exception):
    """
    Process is already running.
    """

class InvalidPIDFileError(Exception):
    """
    PID file contents are invalid.
    """

class StalePIDFileError(Exception):
    """
    PID file contents are valid, but there is no process with the referenced
    PID.
    """

class NoPIDFound(Exception):
    """
    No PID found in PID file.
    """