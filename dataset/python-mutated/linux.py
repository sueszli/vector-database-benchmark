import binascii
import ctypes
import errno
import fcntl
import logging
import select
import socket
import struct
import time
import resource
import tempfile
from abc import ABC, abstractmethod
from functools import partial
from dataclasses import dataclass
from itertools import chain
import io
import os
import random
from elftools.elf.descriptions import describe_symbol_type
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection
from . import linux_syscalls
from .linux_syscall_stubs import SyscallStubs
from ..core.state import TerminateState, Concretize
from ..core.smtlib import ConstraintSet, Operators, Expression, issymbolic, ArrayProxy
from ..core.smtlib.solver import SelectedSolver
from ..exceptions import SolverError
from ..native.cpu.abstractcpu import Cpu, Syscall, ConcretizeArgument, Interruption, Abi, SyscallAbi
from ..native.cpu.cpufactory import CpuFactory
from ..native.memory import SMemory32, SMemory64, Memory32, Memory64, LazySMemory32, LazySMemory64, InvalidMemoryAccess
from ..native.state import State
from ..platforms.platform import Platform, SyscallNotImplemented, unimplemented
from typing import cast, Any, Deque, Dict, IO, Iterable, List, Optional, Set, Tuple, Union, Callable
logger = logging.getLogger(__name__)
MixedSymbolicBuffer = Union[List[Union[bytes, Expression]], bytes]

def errorcode(code: int) -> str:
    if False:
        i = 10
        return i + 15
    return f'errno.{errno.errorcode[code]}'

class RestartSyscall(Exception):
    pass

class Deadlock(Exception):
    pass

class EnvironmentError(RuntimeError):
    pass

class FdError(Exception):

    def __init__(self, message='', err=errno.EBADF):
        if False:
            while True:
                i = 10
        self.err = err
        super().__init__(message)

def perms_from_elf(elf_flags: int) -> str:
    if False:
        return 10
    return ['   ', '  x', ' w ', ' wx', 'r  ', 'r x', 'rw ', 'rwx'][elf_flags & 7]

def perms_from_protflags(prot_flags: int) -> str:
    if False:
        print('Hello World!')
    return ['   ', 'r  ', ' w ', 'rw ', '  x', 'r x', ' wx', 'rwx'][prot_flags & 7]

def mode_from_flags(file_flags: int) -> str:
    if False:
        print('Hello World!')
    return {os.O_RDWR: 'rb+', os.O_RDONLY: 'rb', os.O_WRONLY: 'wb'}[file_flags & 7]

def concreteclass(cls):
    if False:
        while True:
            i = 10
    "\n    This decorator indicates that the given class is intended to have no\n    unimplemented abstract methods.  If this is not the case, a TypeError\n    exception is raised.\n\n    It only really makes sense to use this in conjunction with classes that\n    have ABCMeta as a metaclass, but it should work without issue with other\n    classes too.\n\n    Without using this decorator, instead of getting a TypeError just after\n    class creation time, you will get an error only if you try to instantiate\n    the class.  In short, using this decorator pushes error detection earlier.\n\n    It would be nice if this existed in the Python standard library `abc`\n    module, but it doesn't seem to be present.\n    "
    methods = getattr(cls, '__abstractmethods__', None)
    if methods:
        methods_str = ', '.join((repr(n) for n in sorted(methods)))
        raise TypeError(f'Class {cls.__name__} marked as concrete, but has unimplemented abstract methods: {methods_str}')
    return cls

@dataclass
class StatResult:
    """
    Data structure corresponding to result received from stat, fstat, lstat for
    information about a file.

    See https://man7.org/linux/man-pages/man2/stat.2.html for more info
    """
    st_mode: int
    st_ino: int
    st_dev: int
    st_nlink: int
    st_uid: int
    st_gid: int
    st_size: int
    st_atime: float
    st_mtime: float
    st_ctime: float
    st_blksize: int
    st_blocks: int
    st_rdev: int

def convert_os_stat(stat: os.stat_result) -> StatResult:
    if False:
        i = 10
        return i + 15
    return StatResult(st_mode=stat.st_mode, st_ino=stat.st_ino, st_dev=stat.st_dev, st_nlink=stat.st_nlink, st_uid=stat.st_uid, st_gid=stat.st_gid, st_size=stat.st_size, st_atime=stat.st_atime, st_mtime=stat.st_mtime, st_ctime=stat.st_ctime, st_blksize=stat.st_blksize, st_blocks=stat.st_blocks, st_rdev=stat.st_rdev)

class FdLike(ABC):
    """
    An abstract class for different kinds of file descriptors.
    """

    @abstractmethod
    def read(self, size: int):
        if False:
            while True:
                i = 10
        ...

    @abstractmethod
    def write(self, buf) -> int:
        if False:
            return 10
        ...

    @abstractmethod
    def sync(self) -> None:
        if False:
            return 10
        ...

    @abstractmethod
    def close(self) -> None:
        if False:
            while True:
                i = 10
        ...

    @abstractmethod
    def seek(self, offset: int, whence: int) -> int:
        if False:
            i = 10
            return i + 15
        ...

    @abstractmethod
    def is_full(self) -> bool:
        if False:
            while True:
                i = 10
        ...

    @abstractmethod
    def ioctl(self, request, argp) -> int:
        if False:
            print('Hello World!')
        ...

    @abstractmethod
    def tell(self) -> int:
        if False:
            return 10
        ...

    @abstractmethod
    def stat(self) -> StatResult:
        if False:
            print('Hello World!')
        ...

    @abstractmethod
    def poll(self) -> int:
        if False:
            print('Hello World!')
        ...

    @property
    @abstractmethod
    def closed(self) -> bool:
        if False:
            i = 10
            return i + 15
        ...

@dataclass
class FdTableEntry:
    fdlike: FdLike
    rwaiters: Set[int]
    twaiters: Set[int]

class FdTable:
    """
    This represents Linux's file descriptor table.

    Each file descriptor maps to an C{FdLike} object.  Additionally, each file
    descriptor maps to a set of PIDs of processes that are waiting on that
    descriptor.
    """
    __slots__ = ['_mapping']

    def __init__(self):
        if False:
            return 10
        self._mapping: Dict[int, FdTableEntry] = {}

    def max_fd(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the maximum file descriptor with an entry in this table.\n        '
        return max(self._mapping)

    def _lookup(self, fd: int):
        if False:
            return 10
        try:
            return self._mapping[fd]
        except LookupError:
            raise FdError(f'{fd} is not a valid file descriptor', errno.EBADF)

    def _get_available_fd(self) -> int:
        if False:
            return 10
        m = self._mapping
        num_fds = len(m)
        next_fd = num_fds
        for fd in range(num_fds):
            if fd not in m:
                next_fd = fd
                break
        return next_fd

    def entries(self) -> Iterable[FdTableEntry]:
        if False:
            for i in range(10):
                print('nop')
        return self._mapping.values()

    def has_entry(self, fd: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return fd in self._mapping

    def add_entry(self, f: FdLike) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Adds an entry for the given C{FdLike} to the file descriptor table,\n        returning the file descriptor for it.\n        '
        fd = self._get_available_fd()
        self.add_entry_at(f, fd)
        return fd

    def add_entry_at(self, f: FdLike, fd: int) -> None:
        if False:
            print('Hello World!')
        '\n        Adds an entry for the given C{FdLike} to the file descriptor table at\n        the given file descriptor, which must not already have an entry.\n        '
        assert fd not in self._mapping, f'{fd} already has an entry'
        self._mapping[fd] = FdTableEntry(fdlike=f, rwaiters=set(), twaiters=set())

    def remove_entry(self, fd: int) -> None:
        if False:
            return 10
        if fd not in self._mapping:
            raise FdError(f'{fd} is not a valid file descriptor', errno.EBADF)
        del self._mapping[fd]

    def get_fdlike(self, fd: int) -> FdLike:
        if False:
            i = 10
            return i + 15
        '\n        Returns the C{FdLike} associated with the given file descriptor.\n        Raises C{FdError} if the file descriptor is invalid.\n        '
        return self._lookup(fd).fdlike

    def get_rwaiters(self, fd: int) -> Set[int]:
        if False:
            i = 10
            return i + 15
        return self._lookup(fd).rwaiters

    def get_twaiters(self, fd: int) -> Set[int]:
        if False:
            print('Hello World!')
        return self._lookup(fd).twaiters

@dataclass
class EPollEvent:
    events: int
    data: int

@concreteclass
class EventPoll(FdLike):
    """
    An EventPoll class for epoll support. Internal kernel object referenced by
    a file descriptor
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.interest_list: Dict[FdLike, EPollEvent] = {}

    def read(self, size: int):
        if False:
            print('Hello World!')
        raise NotImplemented

    def write(self, buf) -> int:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplemented

    def sync(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplemented

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def seek(self, offset: int, whence: int) -> int:
        if False:
            i = 10
            return i + 15
        raise NotImplemented

    def is_full(self) -> bool:
        if False:
            i = 10
            return i + 15
        raise NotImplemented

    def ioctl(self, request, argp) -> int:
        if False:
            i = 10
            return i + 15
        raise NotImplemented

    def tell(self) -> int:
        if False:
            while True:
                i = 10
        raise NotImplemented

    def stat(self) -> StatResult:
        if False:
            i = 10
            return i + 15
        raise NotImplemented

    def poll(self) -> int:
        if False:
            return 10
        return select.POLLERR

    def closed(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

@concreteclass
class File(FdLike):

    def __init__(self, path: str, flags: int):
        if False:
            return 10
        mode = mode_from_flags(flags)
        if mode == 'rb+' and (not os.path.exists(path)):
            mode = 'wb+'
        self.file: IO[Any] = open(path, mode)

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        state = {'name': self.name, 'mode': self.mode, 'closed': self.closed}
        try:
            state['pos'] = None if self.closed else self.tell()
        except IOError:
            state['pos'] = None
        return state

    def __setstate__(self, state):
        if False:
            return 10
        name = state['name']
        mode = state['mode']
        closed = state['closed']
        pos = state['pos']
        try:
            self.file = open(name, mode)
            if closed:
                self.file.close()
        except IOError:
            self.file = None
        if pos is not None:
            self.seek(pos)

    @property
    def name(self) -> str:
        if False:
            print('Hello World!')
        return self.file.name

    @property
    def mode(self) -> str:
        if False:
            while True:
                i = 10
        return self.file.mode

    @property
    def closed(self) -> bool:
        if False:
            while True:
                i = 10
        return self.file.closed

    def stat(self) -> StatResult:
        if False:
            i = 10
            return i + 15
        try:
            return convert_os_stat(os.stat(self.fileno()))
        except OSError as e:
            raise FdError(f'Cannot stat: {e.strerror}', e.errno)

    def ioctl(self, request, argp):
        if False:
            return 10
        try:
            return fcntl.fcntl(self, request, argp)
        except OSError as e:
            logger.error(f'Invalid Fcntl request: {request}')
            return -e.errno

    def tell(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.file.tell()

    def seek(self, offset: int, whence: int=os.SEEK_SET) -> int:
        if False:
            i = 10
            return i + 15
        return self.file.seek(offset, whence)

    def pread(self, count, offset):
        if False:
            print('Hello World!')
        return os.pread(self.fileno(), count, offset)

    def write(self, buf):
        if False:
            print('Hello World!')
        return self.file.write(buf)

    def read(self, size):
        if False:
            i = 10
            return i + 15
        return self.file.read(size)

    def close(self) -> None:
        if False:
            return 10
        self.file.close()

    def fileno(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.file.fileno()

    def is_full(self) -> bool:
        if False:
            while True:
                i = 10
        return False

    def sync(self) -> None:
        if False:
            print('Hello World!')
        pass

    def poll(self) -> int:
        if False:
            i = 10
            return i + 15
        'Return EPOLLIN or EPOLLOUT'
        if 'r' in self.mode:
            return select.POLLIN
        elif 'w' in self.mode:
            return select.POLLOUT
        else:
            return select.POLLERR

@concreteclass
class ProcSelfMaps(File):

    def __init__(self, flags: int, linux):
        if False:
            print('Hello World!')
        self.file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.file.write(linux.current.memory.__proc_self__)
        self.file.close()
        mode = mode_from_flags(flags)
        if mode != 'rb':
            raise EnvironmentError('/proc/self/maps is only supported in read only mode')
        self.file = open(self.file.name, mode)

@concreteclass
class Directory(FdLike):

    def __init__(self, path: str, flags: int):
        if False:
            for i in range(10):
                print('nop')
        assert os.path.isdir(path)
        self.fd = os.open(path, flags)
        self.path = path
        self.flags = flags

    def __getstate__(self):
        if False:
            while True:
                i = 10
        state = {}
        state['path'] = self.path
        state['flags'] = self.flags
        return state

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        self.path = state['path']
        self.flags = state['flags']
        self.fd = os.open(self.path, self.flags)

    @property
    def name(self) -> str:
        if False:
            print('Hello World!')
        return self.path

    @property
    def mode(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return mode_from_flags(self.flags)

    @property
    def closed(self) -> bool:
        if False:
            print('Hello World!')
        return False

    def tell(self) -> int:
        if False:
            print('Hello World!')
        return 0

    def seek(self, offset: int, whence: int=os.SEEK_SET) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 0

    def write(self, buf):
        if False:
            for i in range(10):
                print('nop')
        raise FdError('Is a directory', errno.EBADF)

    def read(self, size):
        if False:
            print('Hello World!')
        raise FdError('Is a directory', errno.EISDIR)

    def close(self):
        if False:
            i = 10
            return i + 15
        try:
            return os.close(self.fd)
        except OSError as e:
            return -e.errno

    def stat(self) -> StatResult:
        if False:
            i = 10
            return i + 15
        try:
            return convert_os_stat(os.stat(self.fileno()))
        except OSError as e:
            raise FdError(f'Cannot stat: {e.strerror}', e.errno)

    def fileno(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fd

    def sync(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def is_full(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    def ioctl(self, request, argp):
        if False:
            for i in range(10):
                print('nop')
        raise FdError('Invalid ioctl() operation on Directory', errno.ENOTTY)

    def poll(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return select.POLLERR

@concreteclass
class SymbolicFile(File):
    """
    Represents a symbolic file.
    """

    def __init__(self, constraints, path: str='sfile', flags: int=os.O_RDWR, max_size: int=100, wildcard: str='+'):
        if False:
            i = 10
            return i + 15
        '\n        Builds a symbolic file\n\n        :param constraints: the SMT constraints\n        :param path: the pathname of the symbolic file\n        :param mode: the access permissions of the symbolic file\n        :param max_size: Maximum amount of bytes of the symbolic file\n        :param wildcard: Wildcard to be used in symbolic file\n        '
        super().__init__(path, flags)
        wildcard_buf: bytes = wildcard.encode()
        assert len(wildcard_buf) == 1, f'SymbolicFile wildcard needs to be a single byte, not {wildcard_buf!r}'
        wildcard_val = wildcard_buf[0]
        data = self.file.read()
        self._constraints = constraints
        self.pos = 0
        self.max_size = min(len(data), max_size)
        size = len(data)
        self.array = constraints.new_array(name=self.name, index_max=size)
        symbols_cnt = 0
        for i in range(size):
            if data[i] != wildcard_val:
                self.array[i] = data[i]
            else:
                symbols_cnt += 1
        if symbols_cnt > max_size:
            logger.warning('Found more wildcards in the file than free symbolic values allowed (%d > %d)', symbols_cnt, max_size)
        else:
            logger.debug('Found %d free symbolic values on file %s', symbols_cnt, self.name)

    def __getstate__(self):
        if False:
            return 10
        state = super().__getstate__()
        state['array'] = self.array
        state['pos'] = self.pos
        state['max_size'] = self.max_size
        return state

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        self.pos = state['pos']
        self.max_size = state['max_size']
        self.array = state['array']
        super().__setstate__(state)

    def tell(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the read/write file offset\n        '
        return self.pos

    def seek(self, offset: int, whence: int=os.SEEK_SET) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Repositions the file C{offset} according to C{whence}.\n        Returns the resulting offset or -1 in case of error.\n        :return: the file offset.\n        '
        assert isinstance(offset, int)
        assert whence in (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END)
        new_position = 0
        if whence == os.SEEK_SET:
            new_position = offset
        elif whence == os.SEEK_CUR:
            new_position = self.pos + offset
        elif whence == os.SEEK_END:
            new_position = self.max_size + offset
        if new_position < 0:
            return -1
        self.pos = new_position
        return self.pos

    def read(self, count):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reads up to C{count} bytes from the file.\n        :rtype: list\n        :return: the list of symbolic bytes read\n        '
        if self.pos > self.max_size:
            return []
        else:
            size = min(count, self.max_size - self.pos)
            ret = [self.array[i] for i in range(self.pos, self.pos + size)]
            self.pos += size
            return ret

    def write(self, data):
        if False:
            return 10
        '\n        Writes the symbolic bytes in C{data} onto the file.\n        '
        size = min(len(data), self.max_size - self.pos)
        for i in range(self.pos, self.pos + size):
            self.array[i] = data[i - self.pos]

@concreteclass
class SocketDesc(FdLike):
    """
    Represents a socket descriptor that is not yet connected (i.e. a value returned by socket(2))
    """

    def __init__(self, domain=None, socket_type=None, protocol=None):
        if False:
            for i in range(10):
                print('nop')
        self.domain = domain
        self.socket_type = socket_type
        self.protocol = protocol

    def close(self):
        if False:
            while True:
                i = 10
        pass

    def seek(self, offset: int, whence: int=os.SEEK_SET):
        if False:
            return 10
        raise FdError('Invalid write() operation on SocketDesc', errno.ESPIPE)

    def is_full(self):
        if False:
            for i in range(10):
                print('nop')
        raise IsSocketDescErr()

    def read(self, count):
        if False:
            i = 10
            return i + 15
        raise FdError('Invalid write() operation on SocketDesc', errno.EBADF)

    def write(self, data):
        if False:
            print('Hello World!')
        raise FdError('Invalid write() operation on SocketDesc', errno.EBADF)

    def sync(self):
        if False:
            return 10
        raise FdError('Invalid sync() operation on SocketDesc', errno.EINVAL)

    def ioctl(self, request, argp):
        if False:
            while True:
                i = 10
        raise FdError('Invalid ioctl() operation on SocketDesc', errno.ENOTTY)

    def tell(self) -> int:
        if False:
            while True:
                i = 10
        raise FdError('Invalid tell() operation on SocketDesc', errno.EBADF)

    def stat(self) -> StatResult:
        if False:
            while True:
                i = 10
        return StatResult(8592, 11, 9, 1, 1000, 5, 0, 1378673920, 1378673920, 1378653796, 1024, 34824, 0)

    @property
    def closed(self) -> bool:
        if False:
            while True:
                i = 10
        return False

    def poll(self) -> int:
        if False:
            print('Hello World!')
        return select.POLLERR

@concreteclass
class Socket(FdLike):

    def stat(self):
        if False:
            i = 10
            return i + 15
        return StatResult(8592, 11, 9, 1, 1000, 5, 0, 1378673920, 1378673920, 1378653796, 1024, 34824, 0)

    @staticmethod
    def pair():
        if False:
            print('Hello World!')
        a = Socket()
        b = Socket()
        a.connect(b)
        return (a, b)

    def __init__(self, net: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds a normal socket that does not introduce symbolic bytes.\n\n        :param net: Whether this is a network socket\n        '
        from collections import deque
        self.buffer: Deque[Union[bytes, Expression]] = deque()
        self.peer: Optional[Socket] = None
        self.net: bool = net

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        state = {'buffer': self.buffer, 'net': self.net}
        return state

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.buffer = state['buffer']
        self.net = state['net']

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'SOCKET({hash(self):x}, buffer={self.buffer!r}, net={self.net}, peer={hash(self.peer):x})'

    def is_connected(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.peer is not None

    def is_empty(self) -> bool:
        if False:
            return 10
        return not self.buffer

    def is_full(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return len(self.buffer) > 2 * 1024

    def connect(self, peer):
        if False:
            return 10
        assert not self.is_connected()
        assert not peer.is_connected()
        self.peer = peer
        if peer.peer is None:
            peer.peer = self

    def read(self, size: int):
        if False:
            for i in range(10):
                print('nop')
        return self.receive(size)

    def receive(self, size: int):
        if False:
            i = 10
            return i + 15
        rx_bytes = min(size, len(self.buffer))
        ret = []
        for i in range(rx_bytes):
            ret.append(self.buffer.popleft())
        return ret

    def write(self, buf):
        if False:
            print('Hello World!')
        if self.net:
            return len(buf)
        assert self.is_connected(), f'Non-network socket is not connected: {self.__repr__()}'
        return self.peer._transmit(buf)

    def _transmit(self, buf) -> int:
        if False:
            return 10
        for c in buf:
            self.buffer.append(c)
        return len(buf)

    def sync(self):
        if False:
            for i in range(10):
                print('nop')
        raise FdError('Invalid sync() operation on Socket', errno.EINVAL)

    def seek(self, offset: int, whence: int=os.SEEK_SET) -> int:
        if False:
            print('Hello World!')
        raise FdError('Invalid lseek() operation on Socket', errno.ESPIPE)

    def tell(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        raise FdError('Invalid tell() operation on Socket', errno.EBADF)

    def close(self):
        if False:
            return 10
        '\n        Doesn\'t need to do anything; fixes "no attribute \'close\'" error.\n        '
        pass

    @property
    def closed(self) -> bool:
        if False:
            while True:
                i = 10
        return False

    def ioctl(self, request, argp):
        if False:
            print('Hello World!')
        raise FdError('Invalid ioctl() operation on Socket', errno.ENOTTY)

    def poll(self) -> int:
        if False:
            print('Hello World!')
        if self.is_empty():
            return select.POLLOUT
        return select.POLLIN

@concreteclass
class SymbolicSocket(Socket):
    """
    Symbolic sockets are generally used for network communications that contain user-controlled input.
    """

    def __init__(self, constraints: ConstraintSet, name: str, max_recv_symbolic: int=80, net: bool=True, wildcard: str='+'):
        if False:
            print('Hello World!')
        '\n        Builds a symbolic socket.\n\n        :param constraints: the SMT constraints\n        :param name: The name of the SymbolicSocket, which is propagated to the symbolic variables introduced\n        :param max_recv_symbolic: Maximum number of bytes allowed to be read from this socket. 0 for unlimited\n        :param net: Whether this is a network connection socket\n        :param wildcard: Wildcard to be used for symbolic bytes in socket. Not supported, yet\n        '
        super().__init__(net=net)
        self._constraints = constraints
        self.symb_name = name
        self.max_recv_symbolic = max_recv_symbolic
        self.inputs_recvd: List[ArrayProxy] = []
        self.recv_pos = 0
        self._symb_len: Optional[int] = None
        self.fd: Optional[int] = None

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        state = super().__getstate__()
        state['inputs_recvd'] = self.inputs_recvd
        state['symb_name'] = self.symb_name
        state['recv_pos'] = self.recv_pos
        state['max_recv_symbolic'] = self.max_recv_symbolic
        state['_symb_len'] = self._symb_len
        state['fd'] = self.fd
        return state

    def __setstate__(self, state):
        if False:
            return 10
        super().__setstate__(state)
        self.inputs_recvd = state['inputs_recvd']
        self.symb_name = state['symb_name']
        self.recv_pos = state['recv_pos']
        self.max_recv_symbolic = state['max_recv_symbolic']
        self._symb_len = state['_symb_len']
        self.fd = state['fd']

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'SymbolicSocket({hash(self):x}, fd={self.fd}, inputs_recvd={self.inputs_recvd}, buffer={self.buffer}, net={self.net}'

    def _next_symb_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the next name for a symbolic array, based on previous number of other receives\n        '
        return f'{self.symb_name}-{len(self.inputs_recvd)}'

    def receive(self, size: int) -> Union[ArrayProxy, List[bytes]]:
        if False:
            i = 10
            return i + 15
        '\n        Return a symbolic array of either `size` or rest of remaining symbolic bytes\n        :param size: Size of receive\n        :return: Symbolic array or list of concrete bytes\n        '
        rx_bytes = size if self.max_recv_symbolic == 0 else min(size, self.max_recv_symbolic - self.recv_pos)
        if rx_bytes == 0:
            return []
        if self._symb_len is None:
            self._symb_len = self._constraints.new_bitvec(8, '_socket_symb_len', avoid_collisions=True)
            self._constraints.add(Operators.AND(self._symb_len >= 1, self._symb_len <= rx_bytes))

            def setstate(state: State, value):
                if False:
                    for i in range(10):
                        print('nop')
                state.platform.fd_table.get_fdlike(self.fd)._symb_len = value
            logger.debug('Raising concretize in SymbolicSocket receive')
            raise Concretize('Returning symbolic amount of data to SymbolicSocket', self._symb_len, setstate=setstate, policy='MINMAX')
        ret = self._constraints.new_array(name=self._next_symb_name(), index_max=self._symb_len, avoid_collisions=True)
        logger.info(f'Setting recv symbolic length to {self._symb_len}')
        self.recv_pos += self._symb_len
        self.inputs_recvd.append(ret)
        self._symb_len = None
        return ret

class Linux(Platform):
    """
    A simple Linux Operating System Platform.
    This class emulates the most common Linux system calls
    """
    FCNTL_FDCWD = -100
    BASE_DYN_ADDR_32 = 1448431616
    BASE_DYN_ADDR = 93824992231424

    def __init__(self, program: Optional[str], argv: List[str]=[], envp: List[str]=[], disasm: str='capstone', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds a Linux OS platform\n        :param string program: The path to ELF binary\n        :param string disasm: Disassembler to be used\n        :param list argv: The argv array; not including binary.\n        :param list envp: The ENV variables.\n        '
        super().__init__(path=program, **kwargs)
        self.program = program
        self.clocks: int = 0
        self.fd_table: FdTable = FdTable()
        self._getdents_c: Dict[int, Any] = {}
        self._closed_files: List[FdLike] = []
        self.syscall_trace: List[Tuple[str, int, bytes]] = []
        self.programs = program
        self.disasm = disasm
        self.envp = envp
        self.argv = argv
        self.stubs = SyscallStubs(parent=self)
        self.interp_base: Optional[int] = None
        self.program_base: Optional[int] = None
        self._rlimits = {resource.RLIMIT_NOFILE: (256, 1024), resource.RLIMIT_STACK: (8192 * 1024, 0)}
        if program is not None:
            self.elf = ELFFile(open(program, 'rb'))
            self.arch = {'x86': 'i386', 'x64': 'amd64', 'ARM': 'armv7', 'AArch64': 'aarch64'}[self.elf.get_machine_arch()]
            self._init_cpu(self.arch)
            self._init_std_fds()
            self._execve(program, argv, envp)

    def __del__(self):
        if False:
            print('Hello World!')
        elf = getattr(self, 'elf', None)
        if elf is not None:
            try:
                elf.stream.close()
            except IOError as e:
                logger.error(str(e))

    @property
    def PC(self):
        if False:
            for i in range(10):
                print('nop')
        return (self._current, self.procs[self._current].PC)

    def __deepcopy__(self, memo):
        if False:
            i = 10
            return i + 15
        return self

    @classmethod
    def empty_platform(cls, arch):
        if False:
            print('Hello World!')
        '\n        Create a platform without an ELF loaded.\n\n        :param str arch: The architecture of the new platform\n        :rtype: Linux\n        '
        platform = cls(None)
        platform._init_cpu(arch)
        platform._init_std_fds()
        return platform

    def _init_std_fds(self) -> None:
        if False:
            while True:
                i = 10
        logger.debug('Opening file descriptors (0,1,2) (STDIN, STDOUT, STDERR)')
        self.input = Socket()
        self.output = Socket()
        self.stderr = Socket()
        stdin = Socket()
        stdout = Socket()
        stderr = Socket()
        stdin.peer = self.output
        stdout.peer = self.output
        stderr.peer = self.stderr
        self.input.peer = stdin
        in_fd = self._open(stdin)
        out_fd = self._open(stdout)
        err_fd = self._open(stderr)
        assert (in_fd, out_fd, err_fd) == (0, 1, 2)

    def _init_cpu(self, arch: str) -> None:
        if False:
            return 10
        cpu = self._mk_proc(arch)
        self.procs: List[Cpu] = [cpu]
        self._current: Optional[int] = 0
        self._function_abi = CpuFactory.get_function_abi(cpu, 'linux', arch)
        self._syscall_abi = CpuFactory.get_syscall_abi(cpu, 'linux', arch)

    def _find_symbol(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        symbol_tables = (s for s in self.elf.iter_sections() if isinstance(s, SymbolTableSection))
        for section in symbol_tables:
            if section['sh_entsize'] == 0:
                continue
            for symbol in section.iter_symbols():
                if describe_symbol_type(symbol['st_info']['type']) == 'FUNC':
                    if symbol.name == name:
                        return symbol['st_value']
        return None

    def _execve(self, program: str, argv: List[str], envp: List[str]) -> None:
        if False:
            while True:
                i = 10
        '\n        Load `program` and establish program state, such as stack and arguments.\n\n        :param program str: The ELF binary to load\n        :param argv list: argv array\n        :param envp list: envp array\n        '
        logger.debug(f'Loading {program} as a {self.arch} elf')
        self.load(program, envp)
        self._arch_specific_init()
        self._stack_top = self.current.STACK
        self.setup_stack([program] + argv, envp)
        nprocs = len(self.procs)
        assert nprocs > 0
        self.running = list(range(nprocs))
        self.timers: List[Optional[int]] = [None] * nprocs
        for proc in self.procs:
            self.forward_events_from(proc)

    def _mk_proc(self, arch: str) -> Cpu:
        if False:
            return 10
        mem = Memory32() if arch in {'i386', 'armv7'} else Memory64()
        cpu = CpuFactory.get_cpu(mem, arch)
        return cpu

    @property
    def current(self) -> Cpu:
        if False:
            i = 10
            return i + 15
        assert self._current is not None
        return self.procs[self._current]

    @property
    def function_abi(self) -> Abi:
        if False:
            i = 10
            return i + 15
        assert self._function_abi is not None
        return self._function_abi

    @property
    def syscall_abi(self) -> SyscallAbi:
        if False:
            i = 10
            return i + 15
        assert self._syscall_abi is not None
        return self._syscall_abi

    def __getstate__(self):
        if False:
            print('Hello World!')
        state = super().__getstate__()
        state['clocks'] = self.clocks
        state['input'] = self.input.buffer
        state['output'] = self.output.buffer
        state['fd_table'] = self.fd_table
        state['_getdents_c'] = self._getdents_c
        state['_closed_files'] = self._closed_files
        state['_rlimits'] = self._rlimits
        state['procs'] = self.procs
        state['_current'] = self._current
        state['running'] = self.running
        state['timers'] = self.timers
        state['syscall_trace'] = self.syscall_trace
        state['argv'] = self.argv
        state['envp'] = self.envp
        state['interp_base'] = self.interp_base
        state['program_base'] = self.program_base
        state['elf_bss'] = self.elf_bss
        state['end_code'] = self.end_code
        state['end_data'] = self.end_data
        state['elf_brk'] = self.elf_brk
        state['brk'] = self.brk
        state['auxv'] = self.auxv
        state['program'] = self.program
        state['_function_abi'] = self._function_abi
        state['_syscall_abi'] = self._syscall_abi
        state['_uname_machine'] = self._uname_machine
        _arm_tls_memory = getattr(self, '_arm_tls_memory', None)
        if _arm_tls_memory is not None:
            state['_arm_tls_memory'] = _arm_tls_memory
        return state

    def __setstate__(self, state: Dict) -> None:
        if False:
            i = 10
            return i + 15
        '\n        :todo: some asserts\n        '
        super().__setstate__(state)
        self.input = Socket()
        self.input.buffer = state['input']
        self.output = Socket()
        self.output.buffer = state['output']
        self.fd_table = state['fd_table']
        try:
            stdin = self.fd_table.get_fdlike(0)
            if isinstance(stdin, Socket):
                stdin.peer = self.output
                self.input.peer = stdin
        except FdError:
            pass
        for fd in [1, 2]:
            try:
                f = self.fd_table.get_fdlike(fd)
                if isinstance(f, Socket):
                    f.peer = self.output
            except FdError:
                pass
        self._getdents_c = state['_getdents_c']
        self._closed_files = state['_closed_files']
        self._rlimits = state['_rlimits']
        self.procs = state['procs']
        self._current = state['_current']
        self.running = state['running']
        self.timers = state['timers']
        self.clocks = state['clocks']
        self.syscall_trace = state['syscall_trace']
        self.argv = state['argv']
        self.envp = state['envp']
        self.interp_base = state['interp_base']
        self.program_base = state['program_base']
        self.elf_bss = state['elf_bss']
        self.end_code = state['end_code']
        self.end_data = state['end_data']
        self.elf_brk = state['elf_brk']
        self.brk = state['brk']
        self.auxv = state['auxv']
        self.program = state['program']
        self._function_abi = state['_function_abi']
        self._syscall_abi = state['_syscall_abi']
        self._uname_machine = state['_uname_machine']
        self.stubs = SyscallStubs(parent=self)
        if '_arm_tls_memory' in state:
            self._arm_tls_memory = state['_arm_tls_memory']
        for proc in self.procs:
            self.forward_events_from(proc)

    def _init_arm_kernel_helpers(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        ARM kernel helpers\n\n        https://www.kernel.org/doc/Documentation/arm/kernel_user_helpers.txt\n        '
        page_data = bytearray(b'\xf1\xde\xfd\xe7' * 1024)
        preamble = binascii.unhexlify('ff0300ea' + '650400ea' + 'f0ff9fe5' + '430400ea' + '220400ea' + '810400ea' + '000400ea' + '870400ea')
        __kuser_cmpxchg64 = binascii.unhexlify('30002de9' + '08c09de5' + '30009ce8' + '010055e1' + '00005401' + '0100a013' + '0000a003' + '0c008c08' + '3000bde8' + '1eff2fe1')
        __kuser_dmb = binascii.unhexlify('5bf07ff5' + '1eff2fe1')
        __kuser_cmpxchg = binascii.unhexlify('003092e5' + '000053e1' + '0000a003' + '00108205' + '0100a013' + '1eff2fe1')
        self._arm_tls_memory = self.current.memory.mmap(None, 4, 'rw ')
        __kuser_get_tls = binascii.unhexlify('04009FE5' + '010090e8' + '1eff2fe1') + struct.pack('<I', self._arm_tls_memory)
        tls_area = b'\x00' * 12
        version = struct.pack('<I', 5)

        def update(address, code):
            if False:
                return 10
            page_data[address:address + len(code)] = code
        update(0, preamble)
        update(3936, __kuser_cmpxchg64)
        update(4000, __kuser_dmb)
        update(4032, __kuser_cmpxchg)
        update(4064, __kuser_get_tls)
        update(4080, tls_area)
        update(4092, version)
        self.current.memory.mmap(4294901760, len(page_data), 'r x', page_data)

    def setup_stack(self, argv: List, envp: List) -> None:
        if False:
            i = 10
            return i + 15
        '\n        :param Cpu cpu: The cpu instance\n        :param argv: list of parameters for the program to execute.\n        :param envp: list of environment variables for the program to execute.\n\n        http://www.phrack.org/issues.html?issue=58&id=5#article\n         position            content                     size (bytes) + comment\n         ----------------------------------------------------------------------\n         stack pointer ->  [ argc = number of args ]     4\n                         [ argv[0] (pointer) ]         4   (program name)\n                         [ argv[1] (pointer) ]         4\n                         [ argv[..] (pointer) ]        4 * x\n                         [ argv[n - 1] (pointer) ]     4\n                         [ argv[n] (pointer) ]         4   (= NULL)\n\n                         [ envp[0] (pointer) ]         4\n                         [ envp[1] (pointer) ]         4\n                         [ envp[..] (pointer) ]        4\n                         [ envp[term] (pointer) ]      4   (= NULL)\n\n                         [ auxv[0] (Elf32_auxv_t) ]    8\n                         [ auxv[1] (Elf32_auxv_t) ]    8\n                         [ auxv[..] (Elf32_auxv_t) ]   8\n                         [ auxv[term] (Elf32_auxv_t) ] 8   (= AT_NULL vector)\n\n                         [ padding ]                   0 - 16\n\n                         [ argument ASCIIZ strings ]   >= 0\n                         [ environment ASCIIZ str. ]   >= 0\n\n         (0xbffffffc)      [ end marker ]                4   (= NULL)\n\n         (0xc0000000)      < top of stack >              0   (virtual)\n         ----------------------------------------------------------------------\n        '
        cpu = self.current
        cpu.STACK = self._stack_top
        auxv = self.auxv
        logger.debug('Setting argv, envp and auxv.')
        logger.debug(f'\tArguments: {argv!r}')
        if envp:
            logger.debug('\tEnvironment:')
            for e in envp:
                logger.debug(f'\t\t{e!r}')
        logger.debug('\tAuxv:')
        for (name, val) in auxv.items():
            logger.debug(f'\t\t{name}: 0x{val:x}')
        argvlst = []
        envplst = []
        for evar in envp:
            cpu.push_bytes('\x00')
            envplst.append(cpu.push_bytes(evar))
        for arg in argv:
            cpu.push_bytes('\x00')
            argvlst.append(cpu.push_bytes(arg))
        for (name, value) in auxv.items():
            if hasattr(value, '__len__'):
                cpu.push_bytes(value)
                auxv[name] = cpu.STACK
        auxvnames = {'AT_IGNORE': 1, 'AT_EXECFD': 2, 'AT_PHDR': 3, 'AT_PHENT': 4, 'AT_PHNUM': 5, 'AT_PAGESZ': 6, 'AT_BASE': 7, 'AT_FLAGS': 8, 'AT_ENTRY': 9, 'AT_NOTELF': 10, 'AT_UID': 11, 'AT_EUID': 12, 'AT_GID': 13, 'AT_EGID': 14, 'AT_CLKTCK': 17, 'AT_PLATFORM': 15, 'AT_HWCAP': 16, 'AT_FPUCW': 18, 'AT_SECURE': 23, 'AT_BASE_PLATFORM': 24, 'AT_RANDOM': 25, 'AT_EXECFN': 31, 'AT_SYSINFO': 32, 'AT_SYSINFO_EHDR': 33}
        cpu.push_int(0)
        cpu.push_int(0)
        for (name, val) in auxv.items():
            cpu.push_int(val)
            cpu.push_int(auxvnames[name])
        cpu.push_int(0)
        for var in reversed(envplst):
            cpu.push_int(var)
        envp = cpu.STACK
        cpu.push_int(0)
        for arg in reversed(argvlst):
            cpu.push_int(arg)
        argv = cpu.STACK
        cpu.push_int(len(argvlst))

    def set_entry(self, entryPC):
        if False:
            while True:
                i = 10
        elf_entry = entryPC
        if self.elf.header.e_type == 'ET_DYN':
            elf_entry += self.load_addr
        self.current.PC = elf_entry
        logger.debug(f'Entry point updated: {elf_entry:016x}')

    def load(self, filename: str, env_list: List) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Loads and an ELF program in memory and prepares the initial CPU state.\n        Creates the stack and loads the environment variables and the arguments in it.\n\n        :param filename: pathname of the file to be executed. (used for auxv)\n        :param env_list: A list of env variables. (used for extracting vars that control ld behavior)\n        :raises error:\n            - 'Not matching cpu': if the program is compiled for a different architecture\n            - 'Not matching memory': if the program is compiled for a different address size\n        :todo: define va_randomize and read_implies_exec personality\n        "
        cpu = self.current
        elf = self.elf
        arch = self.arch
        env = dict((var.split('=', 1) for var in env_list if '=' in var))
        addressbitsize = {'x86': 32, 'x64': 64, 'ARM': 32, 'AArch64': 64}[elf.get_machine_arch()]
        logger.debug('Loading %s as a %s elf', filename, arch)
        assert elf.header.e_type in ['ET_DYN', 'ET_EXEC', 'ET_CORE']
        interpreter = None

        def _clean_interp_stream() -> None:
            if False:
                print('Hello World!')
            if interpreter is not None:
                try:
                    interpreter.stream.close()
                except IOError as e:
                    logger.error(str(e))
        for elf_segment in elf.iter_segments():
            if elf_segment.header.p_type != 'PT_INTERP':
                continue
            interpreter_filename = elf_segment.data()[:-1].rstrip(b'\x00').decode('utf-8')
            logger.info(f'Interpreter filename: {interpreter_filename}')
            if 'LD_LIBRARY_PATH' in env:
                for mpath in env['LD_LIBRARY_PATH'].split(':'):
                    interpreter_path_filename = os.path.join(mpath, os.path.basename(interpreter_filename))
                    logger.info(f'looking for interpreter {interpreter_path_filename}')
                    if os.path.exists(interpreter_path_filename):
                        _clean_interp_stream()
                        interpreter = ELFFile(open(interpreter_path_filename, 'rb'))
                        break
            if interpreter is None and os.path.exists(interpreter_filename):
                interpreter = ELFFile(open(interpreter_filename, 'rb'))
            break
        if interpreter is not None:
            assert interpreter.get_machine_arch() == elf.get_machine_arch()
            assert interpreter.header.e_type in ['ET_DYN', 'ET_EXEC']
        executable_stack = False
        for elf_segment in elf.iter_segments():
            if elf_segment.header.p_type != 'PT_GNU_STACK':
                continue
            if elf_segment.header.p_flags & 1:
                executable_stack = True
            else:
                executable_stack = False
            break
        base = 0
        elf_bss = 0
        end_code = 0
        end_data = 0
        elf_brk = 0
        self.load_addr = 0
        for elf_segment in elf.iter_segments():
            if elf_segment.header.p_type != 'PT_LOAD':
                continue
            align = 4096
            ELF_PAGEOFFSET = elf_segment.header.p_vaddr & align - 1
            flags = elf_segment.header.p_flags
            memsz = elf_segment.header.p_memsz + ELF_PAGEOFFSET
            offset = elf_segment.header.p_offset - ELF_PAGEOFFSET
            filesz = elf_segment.header.p_filesz + ELF_PAGEOFFSET
            vaddr = elf_segment.header.p_vaddr - ELF_PAGEOFFSET
            memsz = cpu.memory._ceil(memsz)
            if base == 0 and elf.header.e_type == 'ET_DYN':
                assert vaddr == 0
                if addressbitsize == 32:
                    base = self.BASE_DYN_ADDR_32
                else:
                    base = self.BASE_DYN_ADDR
            perms = perms_from_elf(flags)
            hint = base + vaddr
            if hint == 0:
                hint = None
            logger.debug(f'Loading elf offset: {offset:08x} addr:{base + vaddr:08x} {base + vaddr + memsz:08x} {perms}')
            base = cpu.memory.mmapFile(hint, memsz, perms, elf_segment.stream.name, offset) - vaddr
            if self.load_addr == 0:
                self.load_addr = base + vaddr
            k = base + vaddr + filesz
            if k > elf_bss:
                elf_bss = k
            if flags & 4 and end_code < k:
                end_code = k
            if end_data < k:
                end_data = k
            k = base + vaddr + memsz
            if k > elf_brk:
                elf_brk = k
        elf_entry = elf.header.e_entry
        if elf.header.e_type == 'ET_DYN':
            elf_entry += self.load_addr
        entry = elf_entry
        real_elf_brk = elf_brk
        bytes_to_clear = elf_brk - elf_bss
        if bytes_to_clear > 0:
            logger.debug(f'Zeroing main elf fractional pages. From bss({elf_bss:x}) to brk({elf_brk:x}), {bytes_to_clear} bytes.')
            cpu.write_bytes(elf_bss, '\x00' * bytes_to_clear, force=True)
        stack_size = 135168
        if addressbitsize == 32:
            stack_top = 3221225472
        else:
            stack_top = 140737488355328
        stack_base = stack_top - stack_size
        stack = cpu.memory.mmap(stack_base, stack_size, 'rwx', name='stack') + stack_size
        assert stack_top == stack
        reserved = cpu.memory.mmap(base + vaddr + memsz, 16777216, '   ')
        interpreter_base = 0
        if interpreter is not None:
            base = 0
            elf_bss = 0
            end_code = 0
            end_data = 0
            elf_brk = 0
            entry = interpreter.header.e_entry
            for elf_segment in interpreter.iter_segments():
                if elf_segment.header.p_type != 'PT_LOAD':
                    continue
                align = 4096
                vaddr = elf_segment.header.p_vaddr
                filesz = elf_segment.header.p_filesz
                flags = elf_segment.header.p_flags
                offset = elf_segment.header.p_offset
                memsz = elf_segment.header.p_memsz
                ELF_PAGEOFFSET = vaddr & align - 1
                memsz = memsz + ELF_PAGEOFFSET
                offset = offset - ELF_PAGEOFFSET
                filesz = filesz + ELF_PAGEOFFSET
                vaddr = vaddr - ELF_PAGEOFFSET
                memsz = cpu.memory._ceil(memsz)
                if base == 0 and interpreter.header.e_type == 'ET_DYN':
                    assert vaddr == 0
                    total_size = self._interp_total_size(interpreter)
                    base = stack_base - total_size
                if base == 0:
                    assert vaddr == 0
                perms = perms_from_elf(flags)
                hint = base + vaddr
                if hint == 0:
                    hint = None
                base = cpu.memory.mmapFile(hint, memsz, perms, elf_segment.stream.name, offset)
                base -= vaddr
                logger.debug(f"Loading interpreter offset: {offset:08x} addr:{base + vaddr:08x} {base + vaddr + memsz:08x} {flags & 1 and 'r' or ' '}{flags & 2 and 'w' or ' '}{flags & 4 and 'x' or ' '}")
                k = base + vaddr + filesz
                if k > elf_bss:
                    elf_bss = k
                if flags & 4 and end_code < k:
                    end_code = k
                if end_data < k:
                    end_data = k
                k = base + vaddr + memsz
                if k > elf_brk:
                    elf_brk = k
            if interpreter.header.e_type == 'ET_DYN':
                entry += base
            interpreter_base = base
            bytes_to_clear = elf_brk - elf_bss
            if bytes_to_clear > 0:
                logger.debug(f'Zeroing interpreter elf fractional pages. From bss({elf_bss:x}) to brk({elf_brk:x}), {bytes_to_clear} bytes.')
                cpu.write_bytes(elf_bss, '\x00' * bytes_to_clear, force=True)
        cpu.memory.munmap(reserved, 16777216)
        cpu.STACK = stack
        cpu.PC = entry
        logger.debug(f'Entry point: {entry:016x}')
        logger.debug(f'Stack start: {stack:016x}')
        logger.debug(f'Brk: {real_elf_brk:016x}')
        logger.debug(f'Mappings:')
        for m in str(cpu.memory).split('\n'):
            logger.debug(f'  {m}')
        self.interp_base = base
        self.program_base = self.load_addr
        self.elf_bss = elf_bss
        self.end_code = end_code
        self.end_data = end_data
        self.elf_brk = real_elf_brk
        self.brk = real_elf_brk
        at_random = cpu.push_bytes('A' * 16)
        at_execfn = cpu.push_bytes(f'{filename}\x00')
        self.auxv = {'AT_PHDR': self.load_addr + elf.header.e_phoff, 'AT_PHENT': elf.header.e_phentsize, 'AT_PHNUM': elf.header.e_phnum, 'AT_PAGESZ': cpu.memory.page_size, 'AT_BASE': interpreter_base, 'AT_FLAGS': elf.header.e_flags, 'AT_ENTRY': elf_entry, 'AT_UID': 1000, 'AT_EUID': 1000, 'AT_GID': 1000, 'AT_EGID': 1000, 'AT_CLKTCK': 100, 'AT_HWCAP': 0, 'AT_RANDOM': at_random, 'AT_EXECFN': at_execfn}
        _clean_interp_stream()

    def _to_signed_dword(self, dword: int):
        if False:
            return 10
        arch_width = self.current.address_bit_size
        if arch_width == 32:
            sdword = ctypes.c_int32(dword).value
        elif arch_width == 64:
            sdword = ctypes.c_int64(dword).value
        else:
            raise EnvironmentError(f'Corrupted internal CPU state (arch width is {arch_width})')
        return sdword

    def _open(self, f: FdLike) -> int:
        if False:
            while True:
                i = 10
        '\n        Adds a file descriptor to the current file descriptor list\n\n        :param f: the file descriptor to add.\n        :return: the index of the file descriptor in the file descr. list\n        '
        return self.fd_table.add_entry(f)

    def _close(self, fd: int) -> None:
        if False:
            while True:
                i = 10
        '\n        Removes a file descriptor from the file descriptor list\n        :param fd: the file descriptor to close.\n        '
        f = self.fd_table.get_fdlike(fd)
        f.close()
        self.fd_table.remove_entry(fd)
        self._closed_files.append(f)

    def _is_fd_open(self, fd: int) -> bool:
        if False:
            return 10
        '\n        Determines if the fd is within range and in the file descr. list\n        :param fd: the file descriptor to check.\n        '
        return self.fd_table.has_entry(fd)

    def _get_fdlike(self, fd: int) -> FdLike:
        if False:
            i = 10
            return i + 15
        '\n        Returns the File or Socket corresponding to the given file descriptor.\n        '
        return self.fd_table.get_fdlike(fd)

    def _transform_write_data(self, data) -> bytes:
        if False:
            print('Hello World!')
        '\n        Implement in subclass to transform data written by write(2)/writev(2)\n        Nop by default.\n        '
        return data

    def _exit(self, message) -> None:
        if False:
            for i in range(10):
                print('nop')
        procid = self.procs.index(self.current)
        self.sched()
        self.running.remove(procid)
        if len(self.running) == 0:
            raise TerminateState(message, testcase=True)

    def sys_umask(self, mask: int) -> int:
        if False:
            i = 10
            return i + 15
        '\n        umask - Set file creation mode mask\n        :param int mask: New mask\n        '
        logger.debug(f'umask({mask:o})')
        try:
            return os.umask(mask)
        except OSError as e:
            return -e.errno

    def sys_chdir(self, path) -> int:
        if False:
            return 10
        '\n        chdir - Change current working directory\n        :param int path: Pointer to path\n        '
        path_str = self.current.read_string(path)
        logger.debug(f'chdir({path_str})')
        try:
            os.chdir(path_str)
            return 0
        except OSError as e:
            return -e.errno

    def sys_getcwd(self, buf, size) -> int:
        if False:
            return 10
        '\n        getcwd - Get the current working directory\n        :param int buf: Pointer to dest array\n        :param size: size in bytes of the array pointed to by the buf\n        :return: buf (Success), or 0\n        '
        try:
            current_dir = os.getcwd()
            length = len(current_dir) + 1
            if size > 0 and size < length:
                logger.info('GETCWD: size is greater than 0, but is smaller than the length of the path + 1. Returning -errno.ERANGE')
                return -errno.ERANGE
            if not self.current.memory.access_ok(slice(buf, buf + length), 'w'):
                logger.info('GETCWD: buf within invalid memory. Returning -errno.EFAULT')
                return -errno.EFAULT
            self.current.write_string(buf, current_dir)
            logger.debug(f'getcwd(0x{buf:08x}, {size}) -> <{current_dir}> (Size {length})')
            return length
        except OSError as e:
            return -e.errno

    def sys_lseek(self, fd: int, offset: int, whence: int) -> int:
        if False:
            print('Hello World!')
        '\n        lseek - reposition read/write file offset\n\n        The lseek() function repositions the file offset of the open file description associated\n        with the file descriptor fd to the argument offset according to the directive whence\n\n\n        :param fd: a valid file descriptor\n        :param offset: the offset in bytes\n        :param whence: os.SEEK_SET: The file offset is set to offset bytes.\n                       os.SEEK_CUR: The file offset is set to its current location plus offset bytes.\n                       os.SEEK_END: The file offset is set to the size of the file plus offset bytes.\n\n        :return: offset from file beginning, or EBADF (fd is not a valid file descriptor or is not open)\n        '
        signed_offset = self._to_signed_dword(offset)
        try:
            return self._get_fdlike(fd).seek(signed_offset, whence)
        except FdError as e:
            logger.info(f'sys_lseek: Not valid file descriptor on lseek. Fd not seekable. Returning -{errorcode(e.err)}')
            return -e.err

    def sys_llseek(self, fd: int, offset_high: int, offset_low: int, resultp: int, whence: int) -> int:
        if False:
            while True:
                i = 10
        '\n        _llseek - reposition read/write file offset\n\n        The  _llseek()  system  call  repositions  the  offset  of  the open\n        file description associated with the file descriptor fd to\n        (offset_high<<32) | offset_low bytes relative to the beginning of the\n        file, the current  file offset,  or the end of the file, depending on\n        whether whence is os.SEEK_SET, os.SEEK_CUR, or os.SEEK_END,\n        respectively.  It returns the resulting file position in the argument\n        result.\n\n        This system call exists on various 32-bit platforms to support seeking\n        to large file offsets.\n\n        :param fd: a valid file descriptor\n        :param offset_high: the high 32 bits of the byte offset\n        :param offset_low: the low 32 bits of the byte offset\n        :param resultp: a pointer to write the position into on success\n        :param whence: os.SEEK_SET: The file offset is set to offset bytes.\n                       os.SEEK_CUR: The file offset is set to its current location plus offset bytes.\n                       os.SEEK_END: The file offset is set to the size of the file plus offset bytes.\n\n        :return: 0 on success, negative on error\n        '
        signed_offset_high = self._to_signed_dword(offset_high)
        signed_offset_low = self._to_signed_dword(offset_low)
        signed_offset = signed_offset_high << 32 | signed_offset_low
        try:
            pos = self._get_fdlike(fd).seek(signed_offset, whence)
            posbuf = struct.pack('q', pos)
            self.current.write_bytes(resultp, posbuf)
            return 0
        except FdError as e:
            logger.info(f'sys_llseek: Not valid file descriptor on llseek. Fd not seekable. Returning -{errorcode(e.err)}')
            return -e.err

    def sys_read(self, fd: int, buf: int, count: int) -> int:
        if False:
            return 10
        data: bytes = bytes()
        if count != 0:
            if buf not in self.current.memory:
                logger.info('sys_read: buf points to invalid address. Returning -errno.EFAULT')
                return -errno.EFAULT
            try:
                data = self._get_fdlike(fd).read(count)
            except FdError as e:
                logger.info(f'sys_read: Not valid file descriptor ({fd}). Returning -{errorcode(e.err)}')
                return -e.err
            self.syscall_trace.append(('_read', fd, data))
            self.current.write_bytes(buf, data)
        return len(data)

    def sys_pread64(self, fd: int, buf: int, count: int, offset: int) -> int:
        if False:
            while True:
                i = 10
        '\n        read from a file descriptor at a given offset\n        '
        data: bytes = bytes()
        if count != 0:
            if buf not in self.current.memory:
                logger.info('sys_pread: buf points to invalid address. Returning -errno.EFAULT')
                return -errno.EFAULT
            try:
                target_file = self._get_fdlike(fd)
                if isinstance(target_file, File):
                    data = target_file.pread(count, offset)
                else:
                    logger.error(f'Unsupported pread on {type(target_file)} at fd {fd}')
            except FdError as e:
                logger.info(f'sys_pread: Not valid file descriptor ({fd}). Returning -{errorcode(e.err)}')
                return -e.err
            self.syscall_trace.append(('_pread', fd, data))
            self.current.write_bytes(buf, data)
        return len(data)

    def sys_write(self, fd: int, buf, count) -> int:
        if False:
            return 10
        'write - send bytes through a file descriptor\n        The write system call writes up to count bytes from the buffer pointed\n        to by buf to the file descriptor fd. If count is zero, write returns 0\n        and optionally sets *tx_bytes to zero.\n\n        :param fd            a valid file descriptor\n        :param buf           a memory buffer\n        :param count         number of bytes to send\n        :return: 0          Success\n                  EBADF      fd is not a valid file descriptor or is not open.\n                  EFAULT     buf or tx_bytes points to an invalid address.\n        '
        data: bytes = bytes()
        cpu = self.current
        if count != 0:
            try:
                write_fd = self._get_fdlike(fd)
            except FdError as e:
                logger.error(f'sys_write: Not valid file descriptor ({fd}). Returning -{errorcode(e.err)}')
                return -e.err
            if buf not in cpu.memory or buf + count not in cpu.memory:
                logger.debug('sys_write: buf points to invalid address. Returning -errno.EFAULT')
                return -errno.EFAULT
            if fd > 2 and write_fd.is_full():
                cpu.PC -= cpu.instruction.size
                self.wait([], [fd], None)
                raise RestartSyscall()
            data_sym: MixedSymbolicBuffer = cpu.read_bytes(buf, count)
            data = self._transform_write_data(data_sym)
            write_fd.write(data)
            for line in data.split(b'\n'):
                line_str = line.decode('latin-1')
                logger.debug(f'sys_write({fd}, 0x{buf:08x}, {count}) -> <{repr(line_str):48s}>')
            self.syscall_trace.append(('_write', fd, data))
            self.signal_transmit(fd)
        return len(data)

    def sys_fork(self) -> int:
        if False:
            while True:
                i = 10
        "\n        We don't support forking, but do return a valid error code to client binary.\n        "
        return -errno.ENOSYS
    EPOLLIN = 1
    EPOLLPRI = 2
    EPOLLOUT = 4
    EPOLLERR = 8
    EPOLLHUP = 16
    EPOLLNVAL = 32
    EPOLLRDNORM = 64
    EPOLLRDBAND = 128
    EPOLLWRNORM = 256
    EPOLLWRBAND = 512
    EPOLLMSG = 1024
    EPOLLRDHUP = 8192

    def _do_epoll_create(self, flags: int):
        if False:
            while True:
                i = 10
        EPOLL_CLOEXEC = os.O_CLOEXEC
        if flags & ~EPOLL_CLOEXEC:
            return -errno.EINVAL
        return self.fd_table.add_entry(EventPoll())

    def sys_epoll_create(self, size: int) -> int:
        if False:
            print('Hello World!')
        if size <= 0:
            return -errno.EINVAL
        return self._do_epoll_create(0)

    def sys_epoll_create1(self, flags: int) -> int:
        if False:
            print('Hello World!')
        return self._do_epoll_create(flags)

    def sys_epoll_ctl(self, epfd: int, op: int, fd: int, epds) -> int:
        if False:
            while True:
                i = 10
        "Best effort implementation of what's found in fs/eventpoll.c"
        try:
            epoll_file = self.fd_table.get_fdlike(epfd)
        except FdError:
            return -errno.EBADF
        try:
            target_file = self.fd_table.get_fdlike(fd)
        except FdError:
            return -errno.EBADF
        pass
        if not isinstance(epoll_file, EventPoll) or epoll_file == target_file:
            return -errno.EINVAL
        EPOLL_CTL_ADD = 1
        EPOLL_CTL_DEL = 2
        EPOLL_CTL_MOD = 3
        events = self.current.read_int(epds, size=32)
        data = self.current.read_int(epds + 4, size=64)
        if op == EPOLL_CTL_ADD:
            if target_file in epoll_file.interest_list:
                return -errno.EEXIST
            epoll_file.interest_list[target_file] = EPollEvent(events=events, data=data)
        elif op == EPOLL_CTL_DEL:
            if target_file not in epoll_file.interest_list:
                return -errno.ENOENT
            del epoll_file.interest_list[target_file]
        elif op == EPOLL_CTL_MOD:
            if target_file not in epoll_file.interest_list:
                return -errno.ENOENT
            epoll_file.interest_list[target_file] = EPollEvent(events=events, data=data)
        else:
            return -errno.EINVAL
        return 0

    def _ep_poll(self, ep: EventPoll, events, maxevents: int, timeout) -> int:
        if False:
            while True:
                i = 10
        '\n        Docs from fs/eventpoll.c@ep_poll\n\n        Retrieves ready events, and delivers them to the caller supplied\n        event buffer\n\n        @ep: eventpoll context.\n        @events: Pointer to the userspace buffer where the ready events should be\n                 stored.\n        @maxevents: Size (in terms of number of events) of the caller event buffer.\n        @timeout: Maximum timeout for the ready events fetch operation, in\n                  timespec. If the timeout is zero, the function will not block,\n                  while if the @timeout ptr is NULL, the function will block\n                  until at least one event has been retrieved (or an error\n                  occurred).\n\n        Returns: Returns the number of ready events which have been fetched, or an\n                  error code, in case of error.\n        '
        res: int = 0
        item: FdLike
        einfo: EPollEvent
        removal: Set[FdLike] = set()
        start = time.monotonic()
        while res == 0:
            for (item, einfo) in ep.interest_list.items():
                if item.closed:
                    removal.add(item)
                    continue
                if res >= maxevents:
                    break
                revents: int = einfo.events & item.poll()
                if not bool(revents):
                    continue
                self.current.write_bytes(events, struct.pack('<LQ', revents, einfo.data))
                res += 1
                events += res * 12
            if time.monotonic() - start > timeout / 1000:
                break
        for remove in removal:
            del ep.interest_list[remove]
        return res

    def _do_epoll_wait(self, epfd, events, maxevents, to) -> int:
        if False:
            print('Hello World!')
        sizeof_epoll_event = 12
        EP_MAX_EVENTS = (2 ** 32 - 1) // sizeof_epoll_event
        if maxevents <= 0 or maxevents > EP_MAX_EVENTS:
            return -errno.EINVAL
        if not self.current.memory.access_ok(slice(events, events + sizeof_epoll_event), 'w'):
            return -errno.EFAULT
        try:
            epoll_file = self.fd_table.get_fdlike(epfd)
        except FdError:
            return -errno.EBADF
        if not isinstance(epoll_file, EventPoll):
            return -errno.EINVAL
        return self._ep_poll(epoll_file, events, maxevents, to)

    def sys_epoll_pwait(self, epfd, events, maxevents, timeout, _sigmask, _sigsetsize) -> int:
        if False:
            while True:
                i = 10
        return self._do_epoll_wait(epfd, events, maxevents, timeout)

    def sys_epoll_wait(self, epfd, events, maxevents, timeout) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._do_epoll_wait(epfd, events, maxevents, timeout)

    def sys_access(self, buf: int, mode: int) -> int:
        if False:
            print('Hello World!')
        "\n        Checks real user's permissions for a file\n\n        :param buf: a buffer containing the pathname to the file to check its permissions.\n        :param mode: the access permissions to check.\n        :return:\n            -  C{0} if the calling process can access the file in the desired mode.\n            - C{-1} if the calling process can not access the file in the desired mode.\n        "
        filename = b''
        for i in range(0, 255):
            c = Operators.CHR(self.current.read_int(buf + i, 8))
            if c == b'\x00':
                break
            filename += c
        if os.access(filename, mode):
            return 0
        else:
            if not os.path.exists(filename):
                return -errno.ENOENT
            return -1

    def sys_newuname(self, old_utsname):
        if False:
            print('Hello World!')
        '\n        Writes system information in the variable C{old_utsname}.\n        :rtype: int\n        :param old_utsname: the buffer to write the system info.\n        :return: C{0} on success\n        '
        from datetime import datetime

        def pad(s):
            if False:
                while True:
                    i = 10
            return s + '\x00' * (65 - len(s))
        now = datetime(2017, 8, 1).strftime('%a %b %d %H:%M:%S ART %Y')
        info = (('sysname', 'Linux'), ('nodename', 'ubuntu'), ('release', '4.4.0-77-generic'), ('version', '#98 SMP ' + now), ('machine', self._uname_machine), ('domainname', ''))
        uname_buf = ''.join((pad(pair[1]) for pair in info))
        self.current.write_bytes(old_utsname, uname_buf)
        return 0

    def sys_brk(self, brk):
        if False:
            i = 10
            return i + 15
        '\n        Changes data segment size (moves the C{brk} to the new address)\n        :rtype: int\n        :param brk: the new address for C{brk}.\n        :return: the value of the new C{brk}.\n        :raises error:\n                    - "Error in brk!" if there is any error allocating the memory\n        '
        if brk != 0 and brk > self.elf_brk:
            mem = self.current.memory
            size = brk - self.brk
            if brk > mem._ceil(self.brk):
                perms = mem.perms(self.brk - 1)
                addr = mem.mmap(mem._ceil(self.brk), size, perms)
                if not mem._ceil(self.brk) == addr:
                    logger.error(f'Error in brk: ceil: {hex(mem._ceil(self.brk))} brk: {hex(brk)} self.brk: {hex(self.brk)} addr: {hex(addr)}')
                    return self.brk
            self.brk += size
        return self.brk

    def sys_arch_prctl(self, code, addr):
        if False:
            while True:
                i = 10
        '\n        Sets architecture-specific thread state\n        :rtype: int\n\n        :param code: must be C{ARCH_SET_FS}.\n        :param addr: the base address of the FS segment.\n        :return: C{0} on success\n        :raises error:\n            - if C{code} is different to C{ARCH_SET_FS}\n        '
        ARCH_SET_GS = 4097
        ARCH_SET_FS = 4098
        ARCH_GET_FS = 4099
        ARCH_GET_GS = 4100
        if code not in {ARCH_SET_GS, ARCH_SET_FS, ARCH_GET_FS, ARCH_GET_GS}:
            logger.debug('code not in expected options ARCH_GET/SET_FS/GS')
            return -errno.EINVAL
        if code != ARCH_SET_FS:
            raise NotImplementedError('Manticore supports only arch_prctl with code=ARCH_SET_FS (0x1002) for now')
        self.current.set_descriptor(self.current.FS, addr, 16384, 'rw')
        return 0

    def sys_ioctl(self, fd, request, argp) -> int:
        if False:
            while True:
                i = 10
        if fd > 2:
            try:
                return self.fd_table.get_fdlike(fd).ioctl(request, argp)
            except FdError as e:
                return -e.err
        else:
            return -errno.EINVAL

    def _sys_open_get_file(self, filename: str, flags: int) -> FdLike:
        if False:
            while True:
                i = 10
        if os.path.abspath(filename).startswith('/proc/self'):
            if filename == '/proc/self/exe':
                assert self.program is not None
                filename = os.path.abspath(self.program)
            elif filename == '/proc/self/maps':
                return ProcSelfMaps(flags, self)
            else:
                raise EnvironmentError(f'Trying to read from {filename}.\nThe /proc/self filesystem is largely unsupported.')
        if os.path.isdir(filename):
            return Directory(filename, flags)
        else:
            return File(filename, flags)

    def sys_open(self, buf: int, flags: int, mode: Optional[int]) -> int:
        if False:
            while True:
                i = 10
        '\n        :param buf: address of zero-terminated pathname\n        :param flags: file access bits\n        :param mode: file permission mode (ignored)\n        '
        filename = self.current.read_string(buf)
        try:
            f = self._sys_open_get_file(filename, flags)
            logger.debug(f'sys_open: Opening file {filename} for real file {f!r}')
        except IOError as e:
            logger.warning(f'sys_open: Could not open file {filename}. Reason: {e!s}')
            return -e.errno if e.errno is not None else -errno.EINVAL
        return self._open(f)

    def sys_openat(self, dirfd: int, buf: int, flags: int, mode) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Openat SystemCall - Similar to open system call except dirfd argument\n        when path contained in buf is relative, dirfd is referred to set the relative path\n        Special value AT_FDCWD set for dirfd to set path relative to current directory\n\n        :param dirfd: directory file descriptor to refer in case of relative path at buf\n        :param buf: address of zero-terminated pathname\n        :param flags: file access bits\n        :param mode: file permission mode\n        '
        filename = self.current.read_string(buf)
        dirfd = ctypes.c_int32(dirfd).value
        if os.path.isabs(filename) or dirfd == self.FCNTL_FDCWD:
            return self.sys_open(buf, flags, mode)
        try:
            dir_entry = self._get_fdlike(dirfd)
        except FdError as e:
            logger.info(f'sys_openat: Not valid file descriptor. Returning -{errorcode(e.err)}')
            return -e.err
        if not isinstance(dir_entry, Directory):
            logger.info('sys_openat: Not directory descriptor. Returning -errno.ENOTDIR')
            return -errno.ENOTDIR
        dir_path = dir_entry.name
        filename = os.path.join(dir_path, filename)
        try:
            f = self._sys_open_get_file(filename, flags)
            logger.debug(f'sys_openat: Opening file {filename} for real file {f!r}')
        except IOError as e:
            logger.info(f'sys_openat: Could not open file {filename}. Reason: {e!s}')
            return -e.errno if e.errno is not None else -errno.EINVAL
        return self._open(f)

    def sys_rename(self, oldnamep: int, newnamep: int) -> int:
        if False:
            print('Hello World!')
        '\n        Rename filename `oldnamep` to `newnamep`.\n\n        :param int oldnamep: pointer to oldname\n        :param int newnamep: pointer to newname\n        '
        oldname = self.current.read_string(oldnamep)
        newname = self.current.read_string(newnamep)
        try:
            os.rename(oldname, newname)
        except OSError as e:
            return -e.errno
        return 0

    def sys_fsync(self, fd: int) -> int:
        if False:
            while True:
                i = 10
        "\n        Synchronize a file's in-core state with that on disk.\n        "
        try:
            self._get_fdlike(fd).sync()
            return 0
        except FdError as e:
            return -e.err

    def sys_getpid(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('GETPID, warning pid modeled as concrete 1000')
        return 1000

    def sys_gettid(self):
        if False:
            i = 10
            return i + 15
        logger.debug('GETTID, warning tid modeled as concrete 1000')
        return 1000

    def sys_ARM_NR_set_tls(self, val):
        if False:
            while True:
                i = 10
        if hasattr(self, '_arm_tls_memory'):
            self.current.write_int(self._arm_tls_memory, val)
            self.current.set_arm_tls(val)
        return 0

    def sys_kill(self, pid, sig):
        if False:
            print('Hello World!')
        logger.warning(f'KILL, Ignoring Sending signal {sig} to pid {pid}')
        return 0

    def sys_rt_sigaction(self, signum, act, oldact, _sigsetsize):
        if False:
            print('Hello World!')
        'Wrapper for sys_sigaction'
        return self.sys_sigaction(signum, act, oldact)

    def sys_sigaction(self, signum, act, oldact):
        if False:
            print('Hello World!')
        logger.warning(f'SIGACTION, Ignoring changing signal handler for signal {signum}')
        return 0

    def sys_rt_sigprocmask(self, cpu, how, newset, oldset):
        if False:
            i = 10
            return i + 15
        'Wrapper for sys_sigprocmask'
        return self.sys_sigprocmask(cpu, how, newset, oldset)

    def sys_sigprocmask(self, cpu, how, newset, oldset):
        if False:
            print('Hello World!')
        logger.warning(f'SIGACTION, Ignoring changing signal mask set cmd:%s', how)
        return 0

    def sys_dup(self, fd: int) -> int:
        if False:
            while True:
                i = 10
        '\n        Duplicates an open file descriptor\n        :rtype: int\n        :param fd: the open file descriptor to duplicate.\n        :return: the new file descriptor.\n        '
        try:
            f = self._get_fdlike(fd)
        except FdError as e:
            logger.info(f'sys_dup: fd ({fd}) is not open. Returning -{errorcode(e.err)}')
            return -e.err
        return self._open(f)

    def sys_dup2(self, fd: int, newfd: int) -> int:
        if False:
            print('Hello World!')
        '\n        Duplicates an open fd to newfd. If newfd is open, it is first closed\n        :param fd: the open file descriptor to duplicate.\n        :param newfd: the file descriptor to alias the file described by fd.\n        :return: newfd.\n        '
        try:
            f = self._get_fdlike(fd)
        except FdError as e:
            logger.info('sys_dup2: fd ({fd}) is not open. Returning -{errorcode(e.err)}')
            return -e.err
        (soft_max, hard_max) = self._rlimits[resource.RLIMIT_NOFILE]
        if newfd >= soft_max:
            logger.info(f'sys_dup2: newfd ({newfd}) is above max descriptor table size ({soft_max})')
            return -errno.EBADF
        if self._is_fd_open(newfd):
            self._close(newfd)
        self.fd_table.add_entry_at(f, fd)
        logger.debug('sys_dup2(%d,%d) -> %d', fd, newfd, newfd)
        return newfd

    def sys_chroot(self, path):
        if False:
            i = 10
            return i + 15
        '\n        An implementation of chroot that does perform some basic error checking,\n        but does not actually chroot.\n\n        :param path: Path to chroot\n        '
        if path not in self.current.memory:
            return -errno.EFAULT
        path_s = self.current.read_string(path)
        if not os.path.exists(path_s):
            return -errno.ENOENT
        if not os.path.isdir(path_s):
            return -errno.ENOTDIR
        return -errno.EPERM

    def sys_close(self, fd: int) -> int:
        if False:
            return 10
        '\n        Closes a file descriptor\n        :rtype: int\n        :param fd: the file descriptor to close.\n        :return: C{0} on success.\n        '
        if not self._is_fd_open(fd):
            return -errno.EBADF
        self._close(fd)
        logger.debug(f'sys_close({fd})')
        return 0

    def sys_readlink(self, path, buf, bufsize):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read the value of a symbolic link.\n        :rtype: int\n\n        :param path: symbolic link.\n        :param buf: destination buffer.\n        :param bufsize: size to read.\n        :return: number of bytes placed in buffer on success, -errno on error.\n\n        :todo: return -errno on error.\n        '
        if bufsize <= 0:
            return -errno.EINVAL
        filename = self.current.read_string(path)
        if filename == '/proc/self/exe':
            data = os.path.abspath(self.program)
        else:
            data = os.readlink(filename)[:bufsize]
        self.current.write_bytes(buf, data)
        return len(data)

    def sys_readlinkat(self, dir_fd, path, buf, bufsize):
        if False:
            while True:
                i = 10
        "\n        Read the value of a symbolic link relative to a directory file descriptor.\n        :rtype: int\n\n        :param dir_fd: directory file descriptor.\n        :param path: symbolic link.\n        :param buf: destination buffer.\n        :param bufsize: size to read.\n        :return: number of bytes placed in buffer on success, -errno on error.\n\n        :todo: return -errno on error, full 'dir_fd' support.\n        "
        _path = self.current.read_string(path)
        _dir_fd = ctypes.c_int32(dir_fd).value
        if not (os.path.isabs(_path) or _dir_fd == self.FCNTL_FDCWD):
            raise NotImplementedError('Only absolute paths or paths relative to CWD are supported')
        return self.sys_readlink(path, buf, bufsize)

    def sys_mmap_pgoff(self, address, size, prot, flags, fd, offset):
        if False:
            return 10
        'Wrapper for mmap2'
        return self.sys_mmap2(address, size, prot, flags, fd, offset)

    def sys_mmap2(self, address, size, prot, flags, fd, offset):
        if False:
            i = 10
            return i + 15
        '\n        Creates a new mapping in the virtual address space of the calling process.\n        :rtype: int\n        :param address: the starting address for the new mapping. This address is used as hint unless the\n                        flag contains C{MAP_FIXED}.\n        :param size: the length of the mapping.\n        :param prot: the desired memory protection of the mapping.\n        :param flags: determines whether updates to the mapping are visible to other\n                      processes mapping the same region, and whether updates are carried\n                      through to the underlying file.\n        :param fd: the contents of a file mapping are initialized using C{size} bytes starting at\n                   offset C{offset} in the file referred to by the file descriptor C{fd}.\n        :param offset: the contents of a file mapping are initialized using C{size} bytes starting at\n                       offset C{offset}*0x1000 in the file referred to by the file descriptor C{fd}.\n        :return:\n            - C{-1} In case you use C{MAP_FIXED} in the flags and the mapping can not be place at the desired address.\n            - the address of the new mapping.\n        '
        return self.sys_mmap(address, size, prot, flags, fd, offset * 4096)

    def sys_mmap(self, address, size, prot, flags, fd, offset):
        if False:
            while True:
                i = 10
        '\n        Creates a new mapping in the virtual address space of the calling process.\n        :rtype: int\n\n        :param address: the starting address for the new mapping. This address is used as hint unless the\n                        flag contains C{MAP_FIXED}.\n        :param size: the length of the mapping.\n        :param prot: the desired memory protection of the mapping.\n        :param flags: determines whether updates to the mapping are visible to other\n                      processes mapping the same region, and whether updates are carried\n                      through to the underlying file.\n        :param fd: the contents of a file mapping are initialized using C{size} bytes starting at\n                   offset C{offset} in the file referred to by the file descriptor C{fd}.\n        :param offset: the contents of a file mapping are initialized using C{size} bytes starting at\n                       offset C{offset} in the file referred to by the file descriptor C{fd}.\n        :return:\n                - C{-1} in case you use C{MAP_FIXED} in the flags and the mapping can not be place at the desired address.\n                - the address of the new mapping (that must be the same as address in case you included C{MAP_FIXED} in flags).\n        :todo: handle exception.\n        '
        if address == 0:
            address = None
        cpu = self.current
        if flags & 16:
            cpu.memory.munmap(address, size)
        perms = perms_from_protflags(prot)
        if flags & 32:
            result = cpu.memory.mmap(address, size, perms)
        elif fd == 0:
            assert offset == 0
            result = cpu.memory.mmap(address, size, perms)
            try:
                data = self.fd_table.get_fdlike(fd).read(size)
            except FdError as e:
                return -1
            cpu.write_bytes(result, data)
        else:
            f = self.fd_table.get_fdlike(fd)
            result = cpu.memory.mmapFile(address, size, perms, f.name, offset)
        actually_mapped = f'0x{result:016x}'
        if address is None or result != address:
            address = address or 0
            actually_mapped += f' [requested: 0x{address:016x}]'
        if flags & 16 != 0 and result != address:
            cpu.memory.munmap(result, size)
            result = -1
        return result

    def sys_mprotect(self, start, size, prot):
        if False:
            i = 10
            return i + 15
        "\n        Sets protection on a region of memory. Changes protection for the calling process's\n        memory page(s) containing any part of the address range in the interval [C{start}, C{start}+C{size}-1].\n        :rtype: int\n\n        :param start: the starting address to change the permissions.\n        :param size: the size of the portion of memory to change the permissions.\n        :param prot: the new access permission for the memory.\n        :return: C{0} on success.\n        "
        perms = perms_from_protflags(prot)
        ret = self.current.memory.mprotect(start, size, perms)
        return 0

    def sys_munmap(self, addr, size):
        if False:
            return 10
        '\n        Unmaps a file from memory. It deletes the mappings for the specified address range\n        :rtype: int\n\n        :param addr: the starting address to unmap.\n        :param size: the size of the portion to unmap.\n        :return: C{0} on success.\n        '
        if issymbolic(addr):
            raise ConcretizeArgument(self, 0)
        if issymbolic(size):
            raise ConcretizeArgument(self, 1)
        self.current.memory.munmap(addr, size)
        return 0

    def sys_getuid(self):
        if False:
            return 10
        '\n        Gets user identity.\n        :rtype: int\n\n        :return: this call returns C{1000} for all the users.\n        '
        return 1000

    def sys_getgid(self):
        if False:
            print('Hello World!')
        '\n        Gets group identity.\n        :rtype: int\n\n        :return: this call returns C{1000} for all the groups.\n        '
        return 1000

    def sys_geteuid(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets user identity.\n        :rtype: int\n\n        :return: This call returns C{1000} for all the users.\n        '
        return 1000

    def sys_getegid(self):
        if False:
            print('Hello World!')
        '\n        Gets group identity.\n        :rtype: int\n\n        :return: this call returns C{1000} for all the groups.\n        '
        return 1000

    def sys_readv(self, fd, iov, count) -> int:
        if False:
            print('Hello World!')
        '\n        Works just like C{sys_read} except that data is read into multiple buffers.\n        :rtype: int\n\n        :param fd: the file descriptor of the file to read.\n        :param iov: the buffer where the the bytes to read are stored.\n        :param count: amount of C{iov} buffers to read from the file.\n        :return: the amount of bytes read in total.\n        '
        cpu = self.current
        ptrsize = cpu.address_bit_size
        sizeof_iovec = 2 * (ptrsize // 8)
        total = 0
        for i in range(0, count):
            buf = cpu.read_int(iov + i * sizeof_iovec, ptrsize)
            size = cpu.read_int(iov + i * sizeof_iovec + sizeof_iovec // 2, ptrsize)
            try:
                data = self.fd_table.get_fdlike(fd).read(size)
            except FdError as e:
                return -e.err
            total += len(data)
            cpu.write_bytes(buf, data)
            self.syscall_trace.append(('_read', fd, data))
        return total

    def sys_writev(self, fd, iov, count):
        if False:
            for i in range(10):
                print('nop')
        '\n        Works just like C{sys_write} except that multiple buffers are written out.\n        :rtype: int\n\n        :param fd: the file descriptor of the file to write.\n        :param iov: the buffer where the the bytes to write are taken.\n        :param count: amount of C{iov} buffers to write into the file.\n        :return: the amount of bytes written in total.\n        '
        cpu = self.current
        ptrsize = cpu.address_bit_size
        sizeof_iovec = 2 * (ptrsize // 8)
        total = 0
        try:
            write_fd = self._get_fdlike(fd)
        except FdError as e:
            logger.error(f'writev: Not a valid file descriptor ({fd})')
            return -e.err
        for i in range(0, count):
            buf = cpu.read_int(iov + i * sizeof_iovec, ptrsize)
            size = cpu.read_int(iov + i * sizeof_iovec + sizeof_iovec // 2, ptrsize)
            if issymbolic(size):
                self._publish('will_solve', self.constraints, size, 'get_value')
                size = SelectedSolver.instance().get_value(self.constraints, size)
                self._publish('did_solve', self.constraints, size, 'get_value', size)
            data = [Operators.CHR(cpu.read_int(buf + i, 8)) for i in range(size)]
            data = self._transform_write_data(data)
            write_fd.write(data)
            self.syscall_trace.append(('_write', fd, data))
            total += size
        return total

    def sys_set_thread_area(self, user_info):
        if False:
            print('Hello World!')
        '\n        Sets a thread local storage (TLS) area. Sets the base address of the GS segment.\n        :rtype: int\n\n        :param user_info: the TLS array entry set corresponds to the value of C{u_info->entry_number}.\n        :return: C{0} on success.\n        '
        n = self.current.read_int(user_info, 32)
        pointer = self.current.read_int(user_info + 4, 32)
        m = self.current.read_int(user_info + 8, 32)
        flags = self.current.read_int(user_info + 12, 32)
        assert n == 4294967295
        assert flags == 81
        self.current.GS = 99
        self.current.set_descriptor(self.current.GS, pointer, 16384, 'rw')
        self.current.write_int(user_info, (99 - 3) // 8, 32)
        return 0

    def sys_getpriority(self, which, who):
        if False:
            i = 10
            return i + 15
        '\n        System call ignored.\n        :rtype: int\n\n        :return: C{0}\n        '
        logger.warning('Unimplemented system call: sys_get_priority')
        return 0

    def sys_setpriority(self, which, who, prio):
        if False:
            print('Hello World!')
        '\n        System call ignored.\n        :rtype: int\n\n        :return: C{0}\n        '
        logger.warning('Unimplemented system call: sys_setpriority')
        return 0

    def sys_tgkill(self, tgid, pid, sig):
        if False:
            return 10
        logger.warning('Unimplemented system call: sys_tgkill')
        return 0

    def sys_acct(self, path):
        if False:
            for i in range(10):
                print('nop')
        '\n        System call not implemented.\n        :rtype: int\n\n        :return: C{-1}\n        '
        logger.debug('BSD account not implemented!')
        return -1

    def sys_exit(self, error_code):
        if False:
            print('Hello World!')
        'Wrapper for sys_exit_group'
        return self.sys_exit_group(error_code)

    def sys_exit_group(self, error_code):
        if False:
            print('Hello World!')
        "\n        Exits all threads in a process\n        :raises Exception: 'Finished'\n        "
        return self._exit(f'Program finished with exit status: {ctypes.c_int32(error_code).value}')

    def sys_set_tid_address(self, tidptr):
        if False:
            while True:
                i = 10
        return 1000

    def sys_getrlimit(self, resource, rlim):
        if False:
            for i in range(10):
                print('nop')
        ret = -1
        if resource in self._rlimits:
            rlimit_tup = self._rlimits[resource]
            self.current.write_bytes(rlim, struct.pack('<LL', *rlimit_tup))
            ret = 0
        return ret

    def sys_prlimit64(self, pid, resource, new_lim, old_lim):
        if False:
            print('Hello World!')
        ret = -1
        if pid == 0:
            if old_lim:
                ret = self.sys_getrlimit(resource, old_lim)
            elif new_lim:
                ret = self.sys_setrlimit(resource, new_lim)
        else:
            logger.warning('Cowardly refusing to set resource limits for process %d', pid)
        return ret

    def sys_madvise(self, infop):
        if False:
            return 10
        logger.info('Ignoring sys_madvise')
        return 0

    def sys_fadvise64(self, fd: int, offset: int, length: int, advice: int) -> int:
        if False:
            print('Hello World!')
        logger.info('Ignoring sys_fadvise64')
        return 0

    def sys_arm_fadvise64_64(self, fd: int, offset: int, length: int, advice: int) -> int:
        if False:
            i = 10
            return i + 15
        logger.info('Ignoring sys_arm_fadvise64_64')
        return 0

    def sys_socket(self, domain, socket_type, protocol):
        if False:
            print('Hello World!')
        if domain != socket.AF_INET:
            return -errno.EINVAL
        if socket_type != socket.SOCK_STREAM:
            return -errno.EINVAL
        if protocol != 0:
            return -errno.EINVAL
        f = SocketDesc(domain, socket_type, protocol)
        fd = self._open(f)
        return fd

    def _is_sockfd(self, sockfd: int) -> int:
        if False:
            print('Hello World!')
        try:
            fd = self._get_fdlike(sockfd)
            if not isinstance(fd, SocketDesc):
                return -errno.ENOTSOCK
            return 0
        except IndexError:
            return -errno.EBADF

    def sys_bind(self, sockfd: int, address, address_len) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._is_sockfd(sockfd)

    def sys_listen(self, sockfd: int, backlog) -> int:
        if False:
            i = 10
            return i + 15
        return self._is_sockfd(sockfd)

    def sys_accept(self, sockfd: int, addr, addrlen) -> int:
        if False:
            print('Hello World!')
        '\n        https://github.com/torvalds/linux/blob/63bdf4284c38a48af21745ceb148a087b190cd21/net/socket.c#L1649-L1653\n        '
        return self.sys_accept4(sockfd, addr, addrlen, 0)

    def sys_accept4(self, sockfd: int, addr, addrlen, flags) -> int:
        if False:
            i = 10
            return i + 15
        ret = self._is_sockfd(sockfd)
        if ret != 0:
            return ret
        sock = Socket(net=True)
        fd = self._open(sock)
        return fd

    def sys_recv(self, sockfd: int, buf: int, count: int, flags: int, trace_str='_recv') -> int:
        if False:
            i = 10
            return i + 15
        return self.sys_recvfrom(sockfd, buf, count, flags, 0, 0, trace_str=trace_str)

    def sys_recvfrom(self, sockfd: int, buf: int, count: int, flags: int, src_addr: int, addrlen: int, trace_str='_recvfrom') -> int:
        if False:
            while True:
                i = 10
        if src_addr != 0:
            logger.warning('sys_recvfrom: Unimplemented non-NULL src_addr')
        if addrlen != 0:
            logger.warning('sys_recvfrom: Unimplemented non-NULL addrlen')
        if not self.current.memory.access_ok(slice(buf, buf + count), 'w'):
            logger.info('RECV: buf within invalid memory. Returning -errno.EFAULT')
            return -errno.EFAULT
        try:
            sock = self._get_fdlike(sockfd)
        except FdError:
            return -errno.EBADF
        if not isinstance(sock, Socket):
            return -errno.ENOTSOCK
        data = sock.read(count)
        if len(data) == 0:
            return 0
        self.syscall_trace.append((trace_str, sockfd, data))
        self.current.write_bytes(buf, data)
        return len(data)

    def sys_send(self, sockfd: int, buf: int, count: int, flags: int, trace_str: str='_send') -> int:
        if False:
            for i in range(10):
                print('nop')
        "\n        send(2) is currently a nop; we don't communicate yet: The data is read\n        from memory, but not actually sent anywhere - we just return count to\n        pretend that it was.\n        "
        return self.sys_sendto(sockfd, buf, count, flags, 0, 0, trace_str=trace_str)

    def sys_sendto(self, sockfd: int, buf: int, count: int, flags: int, dest_addr: int, addrlen: int, trace_str: str='_sendto'):
        if False:
            print('Hello World!')
        "\n        sendto(2) is currently a nop; we don't communicate yet: The data is read\n        from memory, but not actually sent anywhere - we just return count to\n        pretend that it was.\n\n        Additionally, dest_addr and addrlen are dropped, so it behaves exactly\n        the same as send.\n        "
        if dest_addr != 0:
            logger.warning('sys_sendto: Unimplemented non-NULL dest_addr')
        if addrlen != 0:
            logger.warning('sys_sendto: Unimplemented non-NULL addrlen')
        try:
            sock = self.fd_table.get_fdlike(sockfd)
        except FdError:
            return -errno.EBADF
        if not isinstance(sock, Socket):
            return -errno.ENOTSOCK
        try:
            data = self.current.read_bytes(buf, count)
        except InvalidMemoryAccess:
            logger.info('SEND: buf within invalid memory. Returning EFAULT')
            return -errno.EFAULT
        self.syscall_trace.append((trace_str, sockfd, data))
        return count

    def sys_sendfile(self, out_fd, in_fd, offset_p, count) -> int:
        if False:
            while True:
                i = 10
        if offset_p != 0:
            offset = self.current.read_int(offset_p, self.current.address_bit_size)
        else:
            offset = 0
        try:
            out_sock = self.fd_table.get_fdlike(out_fd)
            in_sock = self.fd_table.get_fdlike(in_fd)
        except FdError as e:
            return -e.err
        return count

    def sys_getrandom(self, buf, size, flags):
        if False:
            while True:
                i = 10
        "\n        The getrandom system call fills the buffer with random bytes of buflen.\n        The source of random (/dev/random or /dev/urandom) is decided based on\n        the flags value.\n\n        Manticore's implementation simply fills a buffer with zeroes -- choosing\n        determinism over true randomness.\n\n        :param buf: address of buffer to be filled with random bytes\n        :param size: number of random bytes\n        :param flags: source of random (/dev/random or /dev/urandom)\n        :return: number of bytes copied to buf\n        "
        GRND_NONBLOCK = 1
        GRND_RANDOM = 2
        if size == 0:
            return 0
        if buf not in self.current.memory:
            logger.info('getrandom: Provided an invalid address. Returning -errno.EFAULT')
            return -errno.EFAULT
        if flags & ~(GRND_NONBLOCK | GRND_RANDOM):
            return -errno.EINVAL
        self.current.write_bytes(buf, '\x00' * size)
        return size

    @unimplemented
    def sys_futex(self, uaddr, op, val, utime, uaddr2, val3) -> int:
        if False:
            print('Hello World!')
        '\n        Fast user-space locking\n        success: Depends on the operation, but often 0\n        error: Returns -1\n        '
        return 0

    @unimplemented
    def sys_clone_ptregs(self, flags, child_stack, ptid, ctid, regs):
        if False:
            print('Hello World!')
        '\n        Create a child process\n        :param flags:\n        :param child_stack:\n        :param ptid:\n        :param ctid:\n        :param regs:\n        :return: The PID of the child process\n        '
        return self.sys_getpid()

    def syscall(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Syscall dispatcher.\n        '
        index: int = self._syscall_abi.syscall_number()
        name: Optional[str] = None
        try:
            table = getattr(linux_syscalls, self.current.machine)
            name = table.get(index, None)
            if hasattr(self, name):
                implementation = getattr(self, name)
                owner_class = implementation.__qualname__.rsplit('.', 1)[0]
                if owner_class != self.__class__.__name__:
                    implementation = partial(self._handle_unimplemented_syscall, implementation)
            else:
                implementation = getattr(self.stubs, name)
        except (TypeError, AttributeError, KeyError):
            if name is not None:
                raise SyscallNotImplemented(index, name)
            else:
                raise EnvironmentError(f'Bad syscall index, {index}')
        return self._syscall_abi.invoke(implementation)

    def _handle_unimplemented_syscall(self, impl: Callable, *args):
        if False:
            while True:
                i = 10
        '\n        Handle an unimplemented system call (for this class) in a generic way\n        before calling the implementation passed to this function.\n\n        :param impl: The real implementation\n        :param args: The arguments to the implementation\n        '
        return impl(*args)

    def sys_clock_gettime(self, clock_id, timespec):
        if False:
            i = 10
            return i + 15
        logger.warning('sys_clock_time not really implemented')
        if clock_id == 1:
            t = int(time.monotonic() * 1000000000)
            self.current.write_bytes(timespec, struct.pack('L', t // 1000000000) + struct.pack('L', t))
        return 0

    def sys_time(self, tloc):
        if False:
            while True:
                i = 10
        import time
        t = time.time()
        if tloc != 0:
            self.current.write_int(tloc, int(t), self.current.address_bit_size)
        return int(t)

    def sys_gettimeofday(self, tv, tz) -> int:
        if False:
            return 10
        '\n        Get time\n        success: Returns 0\n        error: Returns -1\n        '
        if tv != 0:
            microseconds = int(time.time() * 10 ** 6)
            self.current.write_bytes(tv, struct.pack('L', microseconds // 10 ** 6) + struct.pack('L', microseconds))
        if tz != 0:
            logger.warning('No support for time zones in sys_gettimeofday')
        return 0

    def sched(self) -> None:
        if False:
            while True:
                i = 10
        'Yield CPU.\n        This will choose another process from the running list and change\n        current running process. May give the same cpu if only one running\n        process.\n        '
        if len(self.procs) > 1:
            logger.debug('SCHED:')
            logger.debug(f'\tProcess: {self.procs!r}')
            logger.debug(f'\tRunning: {self.running!r}')
            logger.debug(f'\tTimers: {self.timers!r}')
            logger.debug(f'\tCurrent clock: {self.clocks}')
            logger.debug(f'\tCurrent cpu: {self._current}')
        if len(self.running) == 0:
            logger.debug('None running checking if there is some process waiting for a timeout')
            if all([x is None for x in self.timers]):
                raise Deadlock()
            self.clocks = min((x for x in self.timers if x is not None)) + 1
            self.check_timers()
            assert len(self.running) != 0, 'DEADLOCK!'
            self._current = self.running[0]
            return
        assert self._current is not None
        next_index = (self.running.index(self._current) + 1) % len(self.running)
        next_running_idx = self.running[next_index]
        if len(self.procs) > 1:
            logger.debug(f'\tTransfer control from process {self._current} to {next_running_idx}')
        self._current = next_running_idx

    def wait(self, readfds, writefds, timeout) -> None:
        if False:
            while True:
                i = 10
        'Wait for file descriptors or timeout.\n        Adds the current process in the correspondent waiting list and\n        yield the cpu to another running process.\n        '
        logger.debug('WAIT:')
        logger.debug(f'\tProcess {self._current} is going to wait for [ {readfds!r} {writefds!r} {timeout!r} ]')
        logger.debug(f'\tProcess: {self.procs!r}')
        logger.debug(f'\tRunning: {self.running!r}')
        logger.debug(f'\tTimers: {self.timers!r}')
        assert self._current is not None
        for fd in readfds:
            self.fd_table.get_rwaiters(fd).add(self._current)
        for fd in writefds:
            self.fd_table.get_twaiters(fd).add(self._current)
        if timeout is not None:
            self.timers[self._current] = self.clocks + timeout
        procid = self._current
        next_index = (self.running.index(procid) + 1) % len(self.running)
        self._current = self.running[next_index]
        logger.debug(f'\tTransfer control from process {procid} to {self._current}')
        logger.debug(f'\tREMOVING {procid!r} from {self.running!r}. Current: {self._current!r}')
        self.running.remove(procid)
        if self._current not in self.running:
            logger.debug('\tCurrent not running. Checking for timers...')
            self._current = None
            self.check_timers()

    def awake(self, procid) -> None:
        if False:
            while True:
                i = 10
        'Remove procid from waitlists and reestablish it in the running list'
        logger.debug(f'Remove procid:{procid} from waitlists and reestablish it in the running list')
        for entry in self.fd_table.entries():
            entry.rwaiters.discard(procid)
            entry.twaiters.discard(procid)
        self.timers[procid] = None
        self.running.append(procid)
        if self._current is None:
            self._current = procid

    def connections(self, fd: int) -> Optional[int]:
        if False:
            print('Hello World!')
        'File descriptors are connected to each other like pipes, except\n        for 0, 1, and 2. If you write to FD(N) for N >=3, then that comes\n        out from FD(N+1) and vice-versa\n        '
        if fd in [0, 1, 2]:
            return None
        if fd % 2:
            return fd + 1
        else:
            return fd - 1

    def signal_receive(self, fd: int) -> None:
        if False:
            while True:
                i = 10
        'Awake one process waiting to receive data on fd'
        connections = self.connections
        connection = connections(fd)
        if connection:
            procs = self.fd_table.get_twaiters(connection)
            if procs:
                procid = random.sample(procs, 1)[0]
                self.awake(procid)

    def signal_transmit(self, fd: int) -> None:
        if False:
            i = 10
            return i + 15
        'Awake one process waiting to transmit data on fd'
        connection = self.connections(fd)
        if connection is None or not self.fd_table.has_entry(connection):
            return
        procs = self.fd_table.get_rwaiters(connection)
        if procs:
            procid = random.sample(procs, 1)[0]
            self.awake(procid)

    def check_timers(self) -> None:
        if False:
            while True:
                i = 10
        'Awake process if timer has expired'
        if self._current is None:
            advance = min([self.clocks] + [x for x in self.timers if x is not None]) + 1
            logger.debug(f'Advancing the clock from {self.clocks} to {advance}')
            self.clocks = advance
        for (procid, timer) in enumerate(self.timers):
            if timer is not None:
                if self.clocks > timer:
                    self.procs[procid].PC += self.procs[procid].instruction.size
                    self.awake(procid)

    def execute(self):
        if False:
            print('Hello World!')
        '\n        Execute one cpu instruction in the current thread (only one supported).\n        :rtype: bool\n        :return: C{True}\n\n        :todo: This is where we could implement a simple schedule.\n        '
        try:
            self.current.execute()
            self.clocks += 1
            if self.clocks % 10000 == 0:
                self.check_timers()
                self.sched()
        except (Interruption, Syscall) as e:
            index: int = self._syscall_abi.syscall_number()
            self._syscall_abi._cpu._publish('will_invoke_syscall', index)
            try:
                self.syscall()
                if hasattr(e, 'on_handled'):
                    e.on_handled()
                self._syscall_abi._cpu._publish('did_invoke_syscall', index)
            except RestartSyscall:
                pass
        return True

    def sys_newfstatat(self, dfd, filename, buf, flag):
        if False:
            print('Hello World!')
        '\n        Determines information about a file based on a relative path and a directory file descriptor.\n        :rtype: int\n        :param dfd: directory file descriptor.\n        :param filename: relative path to file.\n        :param buf: a buffer where data about the file will be stored.\n        :param flag: flags to control the query.\n        :return: C{0} on success, negative on error\n        '
        AT_SYMLINK_NOFOLLOW = 256
        AT_EMPTY_PATH = 4096
        dfd = ctypes.c_int32(dfd).value
        flag = ctypes.c_int32(flag).value
        filename_addr = filename
        filename = self.current.read_string(filename, 4096)
        if os.path.isabs(filename) or dfd == self.FCNTL_FDCWD:
            return self.sys_newstat(filename_addr, buf)
        if not len(filename) and flag & AT_EMPTY_PATH:
            return self.sys_newfstat(dfd, buf)
        try:
            f = self._get_fdlike(dfd)
        except FdError as e:
            logger.info(f'sys_newfstatat: invalid fd ({dfd}), returning -{errorcode(e.err)}')
            return -e.err
        if not isinstance(f, Directory):
            return -errno.EISDIR
        follow = not flag & AT_SYMLINK_NOFOLLOW
        try:
            stat = convert_os_stat(os.stat(filename, dir_fd=f.fileno(), follow_symlinks=follow))
        except OSError as e:
            return -e.errno

        def add(width, val):
            if False:
                while True:
                    i = 10
            fformat = {2: 'H', 4: 'L', 8: 'Q'}[width]
            return struct.pack('<' + fformat, val)

        def to_timespec(width, ts):
            if False:
                return 10
            'Note: this is a platform-dependent timespec (8 or 16 bytes)'
            return add(width, int(ts)) + add(width, int(ts % 1 * 1000000000.0))
        nw = self.current.address_bit_size // 8
        bufstat = add(nw, stat.st_dev)
        bufstat += add(nw, stat.st_ino)
        if self.current.address_bit_size == 64:
            bufstat += add(nw, stat.st_nlink)
            bufstat += add(4, stat.st_mode)
            bufstat += add(4, stat.st_uid)
            bufstat += add(4, stat.st_gid)
            bufstat += add(4, 0)
        else:
            bufstat += add(2, stat.st_mode)
            bufstat += add(2, stat.st_nlink)
            bufstat += add(2, stat.st_uid)
            bufstat += add(2, stat.st_gid)
        bufstat += add(nw, stat.st_rdev)
        bufstat += add(nw, stat.st_size)
        bufstat += add(nw, stat.st_blksize)
        bufstat += add(nw, stat.st_blocks)
        bufstat += to_timespec(nw, stat.st_atime)
        bufstat += to_timespec(nw, stat.st_mtime)
        bufstat += to_timespec(nw, stat.st_ctime)
        self.current.write_bytes(buf, bufstat)
        return 0

    def sys_newfstat(self, fd, buf):
        if False:
            while True:
                i = 10
        '\n        Determines information about a file based on its file descriptor.\n        :rtype: int\n        :param fd: the file descriptor of the file that is being inquired.\n        :param buf: a buffer where data about the file will be stored.\n        :return: C{0} on success, EBADF when called with bad fd\n        '
        try:
            stat = self._get_fdlike(fd).stat()
        except FdError as e:
            logger.info(f'sys_newfstat: invalid fd ({fd}), returning -{errorcode(e.err)}')
            return -e.err

        def add(width, val):
            if False:
                i = 10
                return i + 15
            fformat = {2: 'H', 4: 'L', 8: 'Q'}[width]
            return struct.pack('<' + fformat, val)

        def to_timespec(width, ts):
            if False:
                print('Hello World!')
            'Note: this is a platform-dependent timespec (8 or 16 bytes)'
            return add(width, int(ts)) + add(width, int(ts % 1 * 1000000000.0))
        nw = self.current.address_bit_size // 8
        bufstat = add(nw, stat.st_dev)
        bufstat += add(nw, stat.st_ino)
        if self.current.address_bit_size == 64:
            bufstat += add(nw, stat.st_nlink)
            bufstat += add(4, stat.st_mode)
            bufstat += add(4, stat.st_uid)
            bufstat += add(4, stat.st_gid)
            bufstat += add(4, 0)
        else:
            bufstat += add(2, stat.st_mode)
            bufstat += add(2, stat.st_nlink)
            bufstat += add(2, stat.st_uid)
            bufstat += add(2, stat.st_gid)
        bufstat += add(nw, stat.st_rdev)
        bufstat += add(nw, stat.st_size)
        bufstat += add(nw, stat.st_blksize)
        bufstat += add(nw, stat.st_blocks)
        bufstat += to_timespec(nw, stat.st_atime)
        bufstat += to_timespec(nw, stat.st_mtime)
        bufstat += to_timespec(nw, stat.st_ctime)
        self.current.write_bytes(buf, bufstat)
        return 0

    def sys_fstat(self, fd, buf):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines information about a file based on its file descriptor.\n        :rtype: int\n        :param fd: the file descriptor of the file that is being inquired.\n        :param buf: a buffer where data about the file will be stored.\n        :return: C{0} on success, EBADF when called with bad fd\n        '
        try:
            stat = self._get_fdlike(fd).stat()
        except FdError as e:
            logger.info(f'sys_fstat: invalid fd ({fd}), returning -{errorcode(e.err)}')
            return -e.err

        def add(width, val):
            if False:
                return 10
            fformat = {2: 'H', 4: 'L', 8: 'Q'}[width]
            return struct.pack('<' + fformat, val)

        def to_timespec(ts):
            if False:
                for i in range(10):
                    print('nop')
            return struct.pack('<LL', int(ts), int(ts % 1 * 1000000000.0))
        bufstat = add(8, stat.st_dev)
        bufstat += add(4, 0)
        bufstat += add(4, stat.st_ino)
        bufstat += add(4, stat.st_mode)
        bufstat += add(4, stat.st_nlink)
        bufstat += add(4, stat.st_uid)
        bufstat += add(4, stat.st_gid)
        bufstat += add(4, stat.st_rdev)
        bufstat += add(4, stat.st_size)
        bufstat += add(4, stat.st_blksize)
        bufstat += add(4, stat.st_blocks)
        bufstat += to_timespec(stat.st_atime)
        bufstat += to_timespec(stat.st_mtime)
        bufstat += to_timespec(stat.st_ctime)
        bufstat += add(4, 0)
        bufstat += add(4, 0)
        self.current.write_bytes(buf, bufstat)
        return 0

    def sys_fstat64(self, fd, buf):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines information about a file based on its file descriptor (for Linux 64 bits).\n        :rtype: int\n        :param fd: the file descriptor of the file that is being inquired.\n        :param buf: a buffer where data about the file will be stored.\n        :return: C{0} on success, EBADF when called with bad fd\n        :todo: Fix device number.\n        '
        try:
            stat = self._get_fdlike(fd).stat()
        except FdError as e:
            logger.info(f'sys_fstat64: invalid fd ({fd}), returning -{errorcode(e.err)}')
            return -e.err

        def add(width, val):
            if False:
                return 10
            fformat = {2: 'H', 4: 'L', 8: 'Q'}[width]
            return struct.pack('<' + fformat, val)

        def to_timespec(ts):
            if False:
                for i in range(10):
                    print('nop')
            return struct.pack('<LL', int(ts), int(ts % 1 * 1000000000.0))
        bufstat = add(8, stat.st_dev)
        bufstat += add(8, stat.st_ino)
        bufstat += add(4, stat.st_mode)
        bufstat += add(4, stat.st_nlink)
        bufstat += add(4, stat.st_uid)
        bufstat += add(4, stat.st_gid)
        bufstat += add(8, stat.st_rdev)
        bufstat += add(4, 0)
        bufstat += add(8, stat.st_size)
        bufstat += add(4, stat.st_blksize)
        bufstat += add(8, stat.st_blocks)
        bufstat += to_timespec(stat.st_atime)
        bufstat += to_timespec(stat.st_mtime)
        bufstat += to_timespec(stat.st_ctime)
        bufstat += add(4, 0)
        bufstat += add(4, 0)
        self.current.write_bytes(buf, bufstat)
        return 0

    def sys_newstat(self, path, buf):
        if False:
            while True:
                i = 10
        '\n        Wrapper for newfstat()\n        '
        fd = self.sys_open(path, 0, 'r')
        return self.sys_newfstat(fd, buf)

    def sys_stat64(self, path, buf):
        if False:
            print('Hello World!')
        '\n        Determines information about a file based on its filename (for Linux 64 bits).\n        :rtype: int\n        :param path: the pathname of the file that is being inquired.\n        :param buf: a buffer where data about the file will be stored.\n        :return: C{0} on success.\n        '
        return self._stat(path, buf, True)

    def sys_stat32(self, path, buf):
        if False:
            return 10
        return self._stat(path, buf, False)

    def _stat(self, path, buf, is64bit):
        if False:
            while True:
                i = 10
        fd = self.sys_open(path, 0, 'r')
        if is64bit:
            ret = self.sys_fstat64(fd, buf)
        else:
            ret = self.sys_fstat(fd, buf)
        self.sys_close(fd)
        return ret

    def sys_mkdir(self, pathname, mode) -> int:
        if False:
            while True:
                i = 10
        '\n        Creates a directory\n        :return 0 on success\n        '
        name = self.current.read_string(pathname)
        try:
            os.mkdir(name, mode=mode)
        except OSError as e:
            return -e.errno
        return 0

    def sys_rmdir(self, pathname) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Deletes a directory\n        :return 0 on success\n        '
        name = self.current.read_string(pathname)
        try:
            os.rmdir(name)
        except OSError as e:
            return -e.errno
        return -1

    def sys_pipe(self, filedes) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Wrapper for pipe2(filedes, 0)\n        '
        return self.sys_pipe2(filedes, 0)

    def sys_pipe2(self, filedes, flags) -> int:
        if False:
            print('Hello World!')
        '\n        # TODO (ehennenfent) create a native pipe type instead of cheating with sockets\n        Create pipe\n        :return 0 on success\n        '
        if flags == 0:
            (l, r) = Socket.pair()
            self.current.write_int(filedes, self._open(l))
            self.current.write_int(filedes + 4, self._open(r))
            return 0
        else:
            logger.warning("sys_pipe2 doesn't handle flags")
            return -1

    def sys_ftruncate(self, fd, length) -> int:
        if False:
            while True:
                i = 10
        '\n        Truncate a file to <length>\n        :return 0 on success\n        '
        try:
            f = self._get_fdlike(fd)
        except FdError as e:
            return -e.err
        except OSError as e:
            return -e.errno
        if isinstance(f, Directory):
            return -errno.EISDIR
        if not isinstance(f, File):
            return -errno.EINVAL
        f.file.truncate(length)
        return 0

    def sys_link(self, oldname, newname) -> int:
        if False:
            print('Hello World!')
        '\n        Create a symlink from oldname to newname.\n        :return 0 on success\n        '
        oldname = self.current.read_string(oldname)
        newname = self.current.read_string(newname)
        try:
            os.link(oldname, newname)
        except OSError as e:
            return -e.errno
        return 0

    def sys_unlink(self, pathname) -> int:
        if False:
            print('Hello World!')
        '\n        Delete a symlink.\n        :return 0 on success\n        '
        pathname = self.current.read_string(pathname)
        try:
            os.unlink(pathname)
        except OSError as e:
            return -e.errno
        return 0

    def sys_getdents(self, fd, dirent, count) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Fill memory with directory entry structs\n        return: The number of bytes read or 0 at the end of the directory\n        '
        buf = b''
        try:
            file = self._get_fdlike(fd)
        except FdError as e:
            return -e.err
        if not isinstance(file, Directory):
            logger.info("Can't get directory entries for a file")
            return -1
        if fd not in self._getdents_c:
            self._getdents_c[fd] = os.scandir(file.path)
        dent_iter = self._getdents_c[fd]
        item = next(dent_iter, None)
        while item is not None:
            fmt = f'LLH{len(item.name) + 1}sB'
            size = struct.calcsize(fmt)
            if len(buf) + size > count:
                break
            try:
                stat = item.stat()
            except FdError as e:
                return -e.err
            d_type = stat.st_mode >> 12 & 15
            packed = struct.pack(fmt, item.inode(), size, size, bytes(item.name, 'utf-8') + b'\x00', d_type)
            buf += packed
            item = next(dent_iter, None)
        if item:
            self._getdents_c[fd] = chain([item], dent_iter)
        else:
            self._getdents_c[fd] = dent_iter
        if len(buf) > 0:
            self.current.write_bytes(dirent, buf)
        else:
            del self._getdents_c[fd]
        return len(buf)

    def sys_nanosleep(self, rqtp, rmtp) -> int:
        if False:
            return 10
        '\n        Ignored\n        '
        logger.info('Ignoring call to sys_nanosleep')
        return 0

    def sys_chmod(self, filename, mode) -> int:
        if False:
            while True:
                i = 10
        '\n        Modify file permissions\n        :return 0 on success\n        '
        filename = self.current.read_string(filename)
        try:
            os.chmod(filename, mode)
        except OSError as e:
            return -e.errno
        return 0

    def sys_chown(self, filename, user, group) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Modify file ownership\n        :return 0 on success\n        '
        filename = self.current.read_string(filename)
        try:
            os.chown(filename, user, group)
        except OSError as e:
            return -e.errno
        return 0

    def _arch_specific_init(self):
        if False:
            while True:
                i = 10
        assert self.arch in {'i386', 'amd64', 'armv7', 'aarch64'}
        if self.arch == 'i386':
            self._uname_machine = 'i386'
        elif self.arch == 'amd64':
            self._uname_machine = 'x86_64'
        elif self.arch == 'armv7':
            self._uname_machine = 'armv71'
            self._init_arm_kernel_helpers()
            self.current._set_mode_by_val(self.current.PC)
            self.current.PC &= ~1
        elif self.arch == 'aarch64':
            self._uname_machine = 'aarch64'
        if self.arch in {'i386', 'amd64'}:
            x86_defaults = {'CS': 35, 'SS': 43, 'DS': 43, 'ES': 43}
            for (reg, val) in x86_defaults.items():
                self.current.regfile.write(reg, val)

    @staticmethod
    def _interp_total_size(interp):
        if False:
            print('Hello World!')
        '\n        Compute total load size of interpreter.\n\n        :param ELFFile interp: interpreter ELF .so\n        :return: total load size of interpreter, not aligned\n        :rtype: int\n        '
        load_segs = [x for x in interp.iter_segments() if x.header.p_type == 'PT_LOAD']
        last = load_segs[-1]
        return last.header.p_vaddr + last.header.p_memsz

    @classmethod
    def implemented_syscalls(cls) -> Iterable[str]:
        if False:
            return 10
        '\n        Get a listing of all concretely implemented system calls for Linux. This\n        does not include whether a symbolic version exists. To get that listing,\n        use the SLinux.implemented_syscalls() method.\n        '
        import inspect
        return (name for (name, obj) in inspect.getmembers(cls, predicate=inspect.isfunction) if name.startswith('sys_') and getattr(inspect.getmodule(obj), obj.__qualname__.rsplit('.', 1)[0], None) == cls)

    @classmethod
    def unimplemented_syscalls(cls, syscalls: Union[Set[str], Dict[int, str]]) -> Set[str]:
        if False:
            i = 10
            return i + 15
        '\n        Get a listing of all unimplemented concrete system calls for a given\n        collection of Linux system calls. To get a listing of unimplemented\n        symbolic system calls, use the ``SLinux.unimplemented_syscalls()``\n        method.\n\n        Available system calls can be found at ``linux_syscalls.py`` or you may\n        pass your own as either a set of system calls or as a mapping of system\n        call number to system call name.\n\n        Note that passed system calls should follow the naming convention\n        located in ``linux_syscalls.py``.\n        '
        implemented_syscalls = set(cls.implemented_syscalls())
        if isinstance(syscalls, set):
            return syscalls.difference(implemented_syscalls)
        else:
            return set(syscalls.values()).difference(implemented_syscalls)

    @staticmethod
    def print_implemented_syscalls() -> None:
        if False:
            while True:
                i = 10
        for syscall in Linux.implemented_syscalls():
            print(syscall)

class SLinux(Linux):
    """
    Builds a symbolic extension of a Linux OS

    :param str programs: path to ELF binary
    :param str disasm: disassembler to be used
    :param list argv: argv not including binary
    :param list envp: environment variables
    :param tuple[str] symbolic_files: files to consider symbolic
    """

    def __init__(self, programs, argv=None, envp=None, symbolic_files=None, disasm='capstone', pure_symbolic=False):
        if False:
            i = 10
            return i + 15
        argv = [] if argv is None else argv
        envp = [] if envp is None else envp
        symbolic_files = [] if symbolic_files is None else symbolic_files
        self._constraints = ConstraintSet()
        self._pure_symbolic = pure_symbolic
        self.random = 0
        self.symbolic_files = symbolic_files
        self.net_accepts = 0
        super().__init__(programs, argv=argv, envp=envp, disasm=disasm)

    def _mk_proc(self, arch):
        if False:
            print('Hello World!')
        if arch in {'i386', 'armv7'}:
            if self._pure_symbolic:
                mem = LazySMemory32(self.constraints)
            else:
                mem = SMemory32(self.constraints)
        elif self._pure_symbolic:
            mem = LazySMemory64(self.constraints)
        else:
            mem = SMemory64(self.constraints)
        cpu = CpuFactory.get_cpu(mem, arch)
        return cpu

    def add_symbolic_file(self, symbolic_file):
        if False:
            return 10
        "\n        Add a symbolic file. Each '+' in the file will be considered\n        as symbolic; other chars are concretized.\n        Symbolic files must have been defined before the call to `run()`.\n\n        :param str symbolic_file: the name of the symbolic file\n        "
        self.symbolic_files.append(symbolic_file)

    @property
    def constraints(self):
        if False:
            for i in range(10):
                print('nop')
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        if False:
            print('Hello World!')
        self._constraints = constraints
        for proc in self.procs:
            proc.memory.constraints = constraints

    def __getstate__(self):
        if False:
            while True:
                i = 10
        state = super().__getstate__()
        state['constraints'] = self.constraints
        state['random'] = self.random
        state['symbolic_files'] = self.symbolic_files
        state['net_accepts'] = self.net_accepts
        return state

    def __setstate__(self, state):
        if False:
            return 10
        self._constraints = state['constraints']
        self.random = state['random']
        self.symbolic_files = state['symbolic_files']
        self.net_accepts = state['net_accepts']
        super().__setstate__(state)
        for fd_entry in self.fd_table.entries():
            symb_socket_entry = fd_entry.fdlike
            if isinstance(symb_socket_entry, SymbolicSocket):
                symb_socket_entry._constraints = self.constraints

    def _sys_open_get_file(self, filename: str, flags: int) -> FdLike:
        if False:
            i = 10
            return i + 15
        if filename in self.symbolic_files:
            logger.debug(f'{filename} file is considered symbolic')
            return SymbolicFile(self.constraints, filename, flags)
        else:
            return super()._sys_open_get_file(filename, flags)

    def _transform_write_data(self, data: MixedSymbolicBuffer) -> bytes:
        if False:
            print('Hello World!')
        bytes_concretized: int = 0
        concrete_data: bytes = bytes()
        for c in data:
            if issymbolic(c):
                bytes_concretized += 1
                self._publish('will_solve', self.constraints, c, 'get_value')
                c = bytes([SelectedSolver.instance().get_value(self.constraints, c)])
                self._publish('did_solve', self.constraints, c, 'get_value', c)
            concrete_data += cast(bytes, c)
        if bytes_concretized > 0:
            logger.debug(f'Concretized {bytes_concretized} written bytes.')
        return super()._transform_write_data(concrete_data)

    def _handle_unimplemented_syscall(self, impl: Callable, *args):
        if False:
            print('Hello World!')
        '\n        Handle all unimplemented syscalls that could have symbolic arguments.\n\n        If a system call has symbolic argument values and there is no\n        specially implemented function to handle them, then just concretize\n        all symbolic arguments and call impl with args.\n\n        :param name: Name of the system call\n        :param args: Arguments for the system call\n        '
        for (i, arg) in enumerate(args):
            if issymbolic(arg):
                logger.debug(f'Unimplemented symbolic argument to {impl.__name__}. Concretizing argument {i}')
                raise ConcretizeArgument(self, i)
        return impl(*args)

    def sys_exit_group(self, error_code):
        if False:
            for i in range(10):
                print('nop')
        if issymbolic(error_code):
            self._publish('will_solve', self.constraints, error_code, 'get_value')
            error_code = SelectedSolver.instance().get_value(self.constraints, error_code)
            self._publish('did_solve', self.constraints, error_code, 'get_value', error_code)
            return self._exit(f'Program finished with exit status: {ctypes.c_int32(error_code).value} (*)')
        else:
            return super().sys_exit_group(error_code)

    def sys_read(self, fd, buf, count):
        if False:
            return 10
        if issymbolic(fd):
            logger.debug('Ask to read from a symbolic file descriptor!!')
            raise ConcretizeArgument(self, 0)
        if issymbolic(buf):
            logger.debug('Ask to read to a symbolic buffer')
            raise ConcretizeArgument(self, 1)
        if issymbolic(count):
            logger.debug('Ask to read a symbolic number of bytes ')
            raise ConcretizeArgument(self, 2)
        assert not issymbolic(fd)
        assert not issymbolic(buf)
        assert not issymbolic(count)
        return super().sys_read(fd, buf, count)

    def sys_write(self, fd, buf, count):
        if False:
            print('Hello World!')
        if issymbolic(fd):
            logger.debug('Ask to write to a symbolic file descriptor!!')
            raise ConcretizeArgument(self, 0)
        if issymbolic(buf):
            logger.debug('Ask to write to a symbolic buffer')
            raise ConcretizeArgument(self, 1)
        if issymbolic(count):
            logger.debug('Ask to write a symbolic number of bytes ')
            raise ConcretizeArgument(self, 2)
        return super().sys_write(fd, buf, count)

    def sys_recv(self, sockfd, buf, count, flags, trace_str='_recv'):
        if False:
            print('Hello World!')
        if issymbolic(sockfd):
            logger.debug('Ask to read from a symbolic file descriptor!!')
            raise ConcretizeArgument(self, 0)
        if issymbolic(buf):
            logger.debug('Ask to read to a symbolic buffer')
            raise ConcretizeArgument(self, 1)
        if issymbolic(count):
            logger.debug('Ask to read a symbolic number of bytes ')
            raise ConcretizeArgument(self, 2)
        if issymbolic(flags):
            logger.debug('Submitted a symbolic flags')
            raise ConcretizeArgument(self, 3)
        return self.sys_recvfrom(sockfd, buf, count, flags, 0, 0, trace_str)

    def sys_recvfrom(self, sockfd: Union[int, Expression], buf: Union[int, Expression], count: Union[int, Expression], flags: Union[int, Expression], src_addr: Union[int, Expression], addrlen: Union[int, Expression], trace_str: str='_recvfrom'):
        if False:
            return 10
        if issymbolic(sockfd):
            logger.debug('Ask to recvfrom a symbolic file descriptor!!')
            raise ConcretizeArgument(self, 0)
        if issymbolic(buf):
            logger.debug('Ask to recvfrom to a symbolic buffer')
            raise ConcretizeArgument(self, 1)
        if issymbolic(count):
            logger.debug('Ask to recvfrom a symbolic number of bytes ')
            raise ConcretizeArgument(self, 2)
        if issymbolic(flags):
            logger.debug('Ask to recvfrom with symbolic flags')
            raise ConcretizeArgument(self, 3)
        if issymbolic(src_addr):
            logger.debug('Ask to recvfrom with symbolic source address')
            raise ConcretizeArgument(self, 4)
        if issymbolic(addrlen):
            logger.debug('Ask to recvfrom with symbolic address length')
            raise ConcretizeArgument(self, 5)
        assert isinstance(sockfd, int)
        assert isinstance(buf, int)
        assert isinstance(count, int)
        assert isinstance(flags, int)
        assert isinstance(src_addr, int)
        assert isinstance(addrlen, int)
        return super().sys_recvfrom(sockfd, buf, count, flags, src_addr, addrlen, trace_str)

    def sys_accept(self, sockfd, addr, addrlen):
        if False:
            print('Hello World!')
        if issymbolic(sockfd):
            logger.debug('Symbolic sockfd')
            raise ConcretizeArgument(self, 0)
        if issymbolic(addr):
            logger.debug('Symbolic address')
            raise ConcretizeArgument(self, 1)
        if issymbolic(addrlen):
            logger.debug('Symbolic address length')
            raise ConcretizeArgument(self, 2)
        ret = self._is_sockfd(sockfd)
        if ret != 0:
            return ret
        sock = SymbolicSocket(self.constraints, f'SymbSocket_{self.net_accepts}', net=True)
        self.net_accepts += 1
        fd = self._open(sock)
        sock.fd = fd
        return fd

    def sys_open(self, buf: int, flags: int, mode: Optional[int]) -> int:
        if False:
            while True:
                i = 10
        '\n        A version of open(2) that includes a special case for a symbolic path.\n        When given a symbolic path, it will create a temporary file with\n        64 bytes of symbolic bytes as contents and return that instead.\n\n        :param buf: address of zero-terminated pathname\n        :param flags: file access bits\n        :param mode: file permission mode\n        '
        offset = 0
        symbolic_path = issymbolic(self.current.read_int(buf, 8))
        if symbolic_path:
            (fd, path) = tempfile.mkstemp()
            with open(path, 'wb+') as f:
                f.write(b'+' * 64)
            self.symbolic_files.append(path)
            buf = self.current.memory.mmap(None, 1024, 'rw ', data_init=path)
        rv = super().sys_open(buf, flags, mode)
        if symbolic_path:
            self.current.memory.munmap(buf, 1024)
        return rv

    def sys_openat(self, dirfd, buf, flags: int, mode: int) -> int:
        if False:
            while True:
                i = 10
        '\n        A version of openat that includes a symbolic path and symbolic directory file descriptor\n\n        :param dirfd: directory file descriptor\n        :param buf: address of zero-terminated pathname\n        :param flags: file access bits\n        :param mode: file permission mode\n        '
        if issymbolic(dirfd):
            logger.debug('Ask to read from a symbolic directory file descriptor!!')
            self.constraints.add(dirfd >= 0)
            self.constraints.add(dirfd <= (self.fd_table.max_fd() or 0) + 1)
            raise ConcretizeArgument(self, 0)
        if issymbolic(buf):
            logger.debug('Ask to read to a symbolic buffer')
            raise ConcretizeArgument(self, 1)
        return super().sys_openat(dirfd, buf, flags, mode)

    def sys_getrandom(self, buf, size, flags):
        if False:
            i = 10
            return i + 15
        '\n        The getrandom system call fills the buffer with random bytes of buflen.\n        The source of random (/dev/random or /dev/urandom) is decided based on the flags value.\n\n        :param buf: address of buffer to be filled with random bytes\n        :param size: number of random bytes\n        :param flags: source of random (/dev/random or /dev/urandom)\n        :return: number of bytes copied to buf\n        '
        if issymbolic(buf):
            logger.debug('sys_getrandom: Asked to generate random to a symbolic buffer address')
            raise ConcretizeArgument(self, 0)
        if issymbolic(size):
            logger.debug('sys_getrandom: Asked to generate random of symbolic number of bytes')
            raise ConcretizeArgument(self, 1)
        if issymbolic(flags):
            logger.debug('sys_getrandom: Passed symbolic flags')
            raise ConcretizeArgument(self, 2)
        return super().sys_getrandom(buf, size, flags)

    def generate_workspace_files(self) -> Dict[str, Any]:
        if False:
            return 10

        def solve_to_fd(data, fd):
            if False:
                for i in range(10):
                    print('nop')

            def make_chr(c):
                if False:
                    return 10
                if isinstance(c, int):
                    return bytes([c])
                elif isinstance(c, str):
                    return c.encode()
                return c
            try:
                for c in data:
                    if issymbolic(c):
                        self._publish('will_solve', self.constraints, c, 'get_value')
                        c = SelectedSolver.instance().get_value(self.constraints, c)
                        self._publish('did_solve', self.constraints, c, 'get_value', c)
                    fd.write(make_chr(c))
            except SolverError:
                fd.write('{SolverError}')
        out = io.BytesIO()
        inn = io.BytesIO()
        err = io.BytesIO()
        net = io.BytesIO()
        argIO = io.BytesIO()
        envIO = io.BytesIO()
        for (name, fd, data) in self.syscall_trace:
            if name in ('_transmit', '_write'):
                if fd == 1:
                    solve_to_fd(data, out)
                elif fd == 2:
                    solve_to_fd(data, err)
            if name in ('_recv', '_recvfrom'):
                solve_to_fd(data, net)
            if name in ('_receive', '_read') and fd == 0:
                solve_to_fd(data, inn)
        for a in self.argv:
            solve_to_fd(a, argIO)
            argIO.write(b'\n')
        for e in self.envp:
            solve_to_fd(e, envIO)
            envIO.write(b'\n')
        ret = {'syscalls': repr(self.syscall_trace), 'argv': argIO.getvalue(), 'env': envIO.getvalue(), 'stdout': out.getvalue(), 'stdin': inn.getvalue(), 'stderr': err.getvalue(), 'net': net.getvalue()}
        for f in chain((e.fdlike for e in self.fd_table.entries()), self._closed_files):
            if not isinstance(f, SymbolicFile):
                continue
            fdata = io.BytesIO()
            solve_to_fd(f.array, fdata)
            ret[f.name] = fdata.getvalue()
        return ret