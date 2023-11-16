"""
Provides Path, which is analogous to pathlib.Path,
and the type of FSLikeObject.root.
"""
from typing import NoReturn
from io import UnsupportedOperation, TextIOWrapper
import os
import pathlib
import tempfile

class Path:
    """
    Implements an interface somewhat similar to that of pathlib.Path,
    but some methods are missing or have different usage, and some new have
    been added.

    Represents a specific path in a given FS-Like object; mostly, that
    object's member methods are simply wrapped.

    fsobj: fs-like object that is e.g. a cab-archive, a real
           Directory("/lol"), or anything that is like some filesystem.

    parts: starting path in the above fsobj,
           e.g. ["folder", "file"],
           or "folder/file"
           or b"folder/file".
    """

    def __init__(self, fsobj, parts: str | bytes | bytearray | list | tuple=None):
        if False:
            return 10
        if isinstance(parts, str):
            parts = parts.encode()
        if isinstance(parts, (bytes, bytearray)):
            parts = parts.split(b'/')
        if parts is None:
            parts = []
        if not isinstance(parts, (list, tuple)):
            raise ValueError(f'path parts must be str, bytes, list or tuple, but not: {type(parts)}')
        result = []
        for part in parts:
            if isinstance(part, str):
                part = part.encode()
            if part in (b'.', b''):
                pass
            elif part == b'..':
                try:
                    result.pop()
                except IndexError:
                    pass
            else:
                result.append(part)
        self.fsobj = fsobj
        self.is_temp: bool = False
        self.parts = tuple(result)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.fsobj.pretty(self.parts)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if not self.parts:
            return repr(self.fsobj) + '.root'
        return f'Path({repr(self.fsobj)}, {repr(self.parts)})'

    def exists(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ' True if path exists '
        return self.fsobj.exists(self.parts)

    def is_dir(self) -> bool:
        if False:
            i = 10
            return i + 15
        ' True if path points to dir (or symlink to one) '
        return self.fsobj.is_dir(self.parts)

    def is_file(self) -> bool:
        if False:
            i = 10
            return i + 15
        ' True if path points to file (or symlink to one) '
        return self.fsobj.is_file(self.parts)

    def writable(self) -> bool:
        if False:
            while True:
                i = 10
        ' True if path is probably writable '
        return self.fsobj.writable(self.parts)

    def list(self):
        if False:
            print('Hello World!')
        ' Yields path names for all members of this dir '
        yield from self.fsobj.list(self.parts)

    def iterdir(self):
        if False:
            print('Hello World!')
        ' Yields path objects for all members of this dir '
        for name in self.fsobj.list(self.parts):
            yield type(self)(self.fsobj, self.parts + (name,))

    def mkdirs(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Creates this path (including parents). No-op if path exists. '
        return self.fsobj.mkdirs(self.parts)

    def open(self, mode='r'):
        if False:
            return 10
        ' Opens the file at this path; returns a file-like object. '
        dmode = mode.replace('b', '')
        if dmode == 'r':
            handle = self.fsobj.open_r(self.parts)
        elif dmode == 'w':
            handle = self.fsobj.open_w(self.parts)
        elif dmode in ('r+', 'rw'):
            handle = self.fsobj.open_rw(self.parts)
        elif dmode == 'a':
            handle = self.fsobj.open_a(self.parts)
        elif dmode in ('a+', 'ar'):
            handle = self.fsobj.open_ar(self.parts)
        else:
            raise UnsupportedOperation('unsupported open mode: ' + mode)
        if handle is None:
            raise IOError(f'failed to acquire valid file handle for {self} in mode {mode}')
        if 'b' in mode:
            return handle
        return TextIOWrapper(handle)

    def open_r(self):
        if False:
            for i in range(10):
                print('nop')
        " open with mode='rb' "
        return self.fsobj.open_r(self.parts)

    def open_w(self):
        if False:
            for i in range(10):
                print('nop')
        " open with mode='wb' "
        return self.fsobj.open_w(self.parts)

    def open_a(self):
        if False:
            i = 10
            return i + 15
        " open with mode='ab' "
        return self.fsobj.open_a(self.parts)

    def _get_native_path(self):
        if False:
            while True:
                i = 10
        "\n        return the native path (usable by your kernel) of this path,\n        or None if the path is not natively usable.\n\n        Don't use this method directly, use the resolve methods below.\n        "
        return self.fsobj.get_native_path(self.parts)

    def _resolve_r(self):
        if False:
            i = 10
            return i + 15
        '\n        Flatten the path recursively for read access.\n        Used to cancel out some wrappers in between.\n        '
        return self.fsobj.resolve_r(self.parts)

    def _resolve_w(self):
        if False:
            print('Hello World!')
        '\n        Flatten the path recursively for write access.\n        Used to cancel out some wrappers in between.\n        '
        return self.fsobj.resolve_w(self.parts)

    def resolve_native_path(self, mode='r'):
        if False:
            while True:
                i = 10
        '\n        Minimize the path and possibly return a native one.\n        Returns None if there was no native path.\n        '
        if mode == 'r':
            return self.resolve_native_path_r()
        if mode == 'w':
            return self.resolve_native_path_w()
        raise UnsupportedOperation('unsupported resolve mode: ' + mode)

    def resolve_native_path_r(self):
        if False:
            i = 10
            return i + 15
        '\n        Resolve the path for read access and possibly return\n        a native equivalent.\n        If no native path was found, return None.\n        '
        resolved_path = self._resolve_r()
        if resolved_path:
            return resolved_path._get_native_path()
        return None

    def resolve_native_path_w(self):
        if False:
            i = 10
            return i + 15
        '\n        Resolve the path for write access and try to return\n        a native equivalent.\n        If no native path could be determined, return None.\n        '
        resolved_path = self._resolve_w()
        if resolved_path:
            return resolved_path._get_native_path()
        return None

    def rename(self, targetpath):
        if False:
            while True:
                i = 10
        ' renames to targetpath '
        if self.fsobj != targetpath.fsobj:
            raise UnsupportedOperation("can't rename across two FSLikeObjects")
        return self.fsobj.rename(self.parts, targetpath.parts)

    def rmdir(self):
        if False:
            return 10
        ' Removes the empty directory at this path. '
        return self.fsobj.rmdir(self.parts)

    def touch(self):
        if False:
            for i in range(10):
                print('nop')
        ' Creates the file at this path, or updates the timestamp. '
        return self.fsobj.touch(self.parts)

    def unlink(self):
        if False:
            for i in range(10):
                print('nop')
        ' Removes the file at this path. '
        return self.fsobj.unlink(self.parts)

    def removerecursive(self):
        if False:
            while True:
                i = 10
        ' Recursively deletes this file or directory. '
        if self.is_dir():
            for path in self.iterdir():
                path.removerecursive()
            self.rmdir()
        else:
            self.unlink()

    @property
    def mtime(self):
        if False:
            i = 10
            return i + 15
        ' Returns the time of last modification of the file or directory. '
        return self.fsobj.mtime(self.parts)

    @property
    def filesize(self):
        if False:
            i = 10
            return i + 15
        ' Returns the file size. '
        return self.fsobj.filesize(self.parts)

    def watch(self, callback):
        if False:
            i = 10
            return i + 15
        "\n        Installs 'callback' as callback that gets invoked whenever the file at\n        this path changes.\n\n        Returns True if the callback was installed, and false if not\n        (e.g. because the some OS limit was reached, or the underlying\n         FSLikeObject doesn't support watches).\n        "
        return self.fsobj.watch(self.parts, callback)

    def poll_fs_watches(self):
        if False:
            return 10
        ' Polls the installed watches for the entire file-system. '
        self.fsobj.poll_watches()

    @property
    def parent(self):
        if False:
            while True:
                i = 10
        ' Parent path object. The parent of root is root. '
        return type(self)(self.fsobj, self.parts[:-1])

    @property
    def name(self):
        if False:
            print('Hello World!')
        ' The name of the topmost component (str). '
        return self.parts[-1].decode()

    @property
    def suffix(self):
        if False:
            i = 10
            return i + 15
        ' The last suffix of the name of the topmost component (str). '
        name = self.name
        pos = name.rfind('.')
        if pos <= 0:
            return ''
        return name[pos:]

    @property
    def suffixes(self):
        if False:
            i = 10
            return i + 15
        ' The suffixes of the name of the topmost component (str list). '
        name = self.name
        if name.startswith('.'):
            name = name[1:]
        return ['.' + suffix for suffix in name.split('.')[1:]]

    @property
    def stem(self):
        if False:
            return 10
        ' Name without suffix (such that stem + suffix == name). '
        name = self.name
        pos = name.rfind('.')
        if pos <= 0:
            return name
        return name[:pos]

    def joinpath(self, subpath):
        if False:
            while True:
                i = 10
        ' Returns path for the given subpath. '
        if isinstance(subpath, str):
            subpath = subpath.encode()
        if isinstance(subpath, bytes):
            subpath = subpath.split(b'/')
        return type(self)(self.fsobj, self.parts + tuple(subpath))

    def __getitem__(self, subpath):
        if False:
            return 10
        ' Like joinpath. '
        return self.joinpath(subpath)

    def __truediv__(self, subpath):
        if False:
            i = 10
            return i + 15
        ' Like joinpath. '
        return self.joinpath(subpath)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        ' comparison by fslike and parts '
        return self.fsobj == other.fsobj and self.parts == other.parts

    def with_name(self, name):
        if False:
            while True:
                i = 10
        ' Returns path for differing name (same parent). '
        return self.parent.joinpath(name)

    def with_suffix(self, suffix):
        if False:
            for i in range(10):
                print('nop')
        ' Returns path for different suffix (same parent and stem). '
        if isinstance(suffix, bytes):
            suffix = suffix.decode()
        return self.parent.joinpath(self.stem + suffix)

    def mount(self, pathobj, priority=0) -> NoReturn:
        if False:
            print('Hello World!')
        "This is only valid for UnionPath, don't call here"
        raise PermissionError('Do not call mount on Path instances!')

    @staticmethod
    def get_temp_file():
        if False:
            return 10
        '\n        Creates a temporary file.\n        '
        (temp_fd, temp_file) = tempfile.mkstemp()
        os.close(temp_fd)
        path = Path(pathlib.Path(temp_file))
        path.is_temp = True
        return path

    @staticmethod
    def get_temp_dir():
        if False:
            while True:
                i = 10
        '\n        Creates a temporary directory.\n        '
        temp_dir = tempfile.mkdtemp()
        path = Path(pathlib.Path(temp_dir))
        path.is_temp = True
        return path