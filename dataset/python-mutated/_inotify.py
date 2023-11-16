"""
Very low-level ctypes-based interface to Linux inotify(7).

ctypes and a version of libc which supports inotify system calls are
required.
"""
import ctypes
import ctypes.util
from typing import Any, cast
from twisted.python.filepath import FilePath

class INotifyError(Exception):
    """
    Unify all the possible exceptions that can be raised by the INotify API.
    """

def init() -> int:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create an inotify instance and return the associated file descriptor.\n    '
    fd = cast(int, libc.inotify_init())
    if fd < 0:
        raise INotifyError('INotify initialization error.')
    return fd

def add(fd: int, path: FilePath[Any], mask: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    '\n    Add a watch for the given path to the inotify file descriptor, and return\n    the watch descriptor.\n\n    @param fd: The file descriptor returned by C{libc.inotify_init}.\n    @param path: The path to watch via inotify.\n    @param mask: Bitmask specifying the events that inotify should monitor.\n    '
    wd = cast(int, libc.inotify_add_watch(fd, path.asBytesMode().path, mask))
    if wd < 0:
        raise INotifyError(f"Failed to add watch on '{path!r}' - ({wd!r})")
    return wd

def remove(fd: int, wd: int) -> None:
    if False:
        while True:
            i = 10
    '\n    Remove the given watch descriptor from the inotify file descriptor.\n    '
    libc.inotify_rm_watch(fd, wd)

def initializeModule(libc: ctypes.CDLL) -> None:
    if False:
        print('Hello World!')
    '\n    Initialize the module, checking if the expected APIs exist and setting the\n    argtypes and restype for C{inotify_init}, C{inotify_add_watch}, and\n    C{inotify_rm_watch}.\n    '
    for function in ('inotify_add_watch', 'inotify_init', 'inotify_rm_watch'):
        if getattr(libc, function, None) is None:
            raise ImportError('libc6 2.4 or higher needed')
    libc.inotify_init.argtypes = []
    libc.inotify_init.restype = ctypes.c_int
    libc.inotify_rm_watch.argtypes = [ctypes.c_int, ctypes.c_int]
    libc.inotify_rm_watch.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int
name = ctypes.util.find_library('c')
if not name:
    raise ImportError("Can't find C library.")
libc = ctypes.cdll.LoadLibrary(name)
initializeModule(libc)