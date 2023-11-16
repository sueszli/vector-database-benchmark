"""
FSLikeObjects that represent actual file system paths:

 - Directory: enforces case
 - CaseIgnoringReadOnlyDirectory
"""
from __future__ import annotations
import typing
import os
import pathlib
from typing import Union
from .abstract import FSLikeObject
if typing.TYPE_CHECKING:
    from io import BufferedReader

class Directory(FSLikeObject):
    """
    Provides an actual file system directory's contents as-they-are.

    Initialized from some real path that is mounted already by your system.
    """

    def __init__(self, path_, create_if_missing=False):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(path_, pathlib.Path):
            path = bytes(path_)
        elif isinstance(path_, str):
            path = path_.encode()
        elif isinstance(path_, bytes):
            path = path_
        else:
            raise TypeError(f'incompatible type for path: {type(path_)}')
        if not os.path.isdir(path):
            if create_if_missing:
                os.makedirs(path)
            else:
                raise FileNotFoundError(path)
        self.path = path

    def __repr__(self):
        if False:
            print('Hello World!')
        return f"Directory({self.path.decode(errors='replace')})"

    def resolve(self, parts) -> Union[str, bytes]:
        if False:
            return 10
        ' resolves parts to an actual path name. '
        return os.path.join(self.path, *parts)

    def open_r(self, parts) -> BufferedReader:
        if False:
            return 10
        return open(self.resolve(parts), 'rb')

    def open_w(self, parts) -> BufferedReader:
        if False:
            while True:
                i = 10
        return open(self.resolve(parts), 'wb')

    def open_rw(self, parts) -> BufferedReader:
        if False:
            return 10
        return open(self.resolve(parts), 'r+b')

    def open_a(self, parts) -> BufferedReader:
        if False:
            i = 10
            return i + 15
        return open(self.resolve(parts), 'ab')

    def open_ar(self, parts) -> BufferedReader:
        if False:
            print('Hello World!')
        return open(self.resolve(parts), 'a+b')

    def get_native_path(self, parts) -> Union[str, bytes]:
        if False:
            while True:
                i = 10
        return self.resolve(parts)

    def list(self, parts) -> typing.Generator[str | bytes, None, None]:
        if False:
            for i in range(10):
                print('nop')
        yield from os.listdir(self.resolve(parts))

    def filesize(self, parts) -> int:
        if False:
            for i in range(10):
                print('nop')
        return os.path.getsize(self.resolve(parts))

    def mtime(self, parts) -> float:
        if False:
            print('Hello World!')
        return os.path.getmtime(self.resolve(parts))

    def mkdirs(self, parts) -> None:
        if False:
            while True:
                i = 10
        return os.makedirs(self.resolve(parts), exist_ok=True)

    def rmdir(self, parts) -> None:
        if False:
            for i in range(10):
                print('nop')
        return os.rmdir(self.resolve(parts))

    def unlink(self, parts) -> None:
        if False:
            print('Hello World!')
        return os.unlink(self.resolve(parts))

    def touch(self, parts) -> None:
        if False:
            return 10
        try:
            os.utime(self.resolve(parts))
        except FileNotFoundError:
            with open(self.resolve(parts), 'ab') as directory:
                directory.close()

    def rename(self, srcparts, tgtparts) -> None:
        if False:
            return 10
        return os.rename(self.resolve(srcparts), self.resolve(tgtparts))

    def is_file(self, parts) -> bool:
        if False:
            i = 10
            return i + 15
        return os.path.isfile(self.resolve(parts))

    def is_dir(self, parts) -> bool:
        if False:
            i = 10
            return i + 15
        return os.path.isdir(self.resolve(parts))

    def writable(self, parts) -> bool:
        if False:
            return 10
        parts = list(parts)
        path = self.resolve(parts)
        while not os.path.exists(path):
            if not parts:
                raise FileNotFoundError(self.path)
            parts.pop()
            path = self.resolve(parts)
        return os.access(path, os.W_OK)

    def watch(self, parts, callback) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def poll_watches(self) -> None:
        if False:
            while True:
                i = 10
        pass

class CaseIgnoringDirectory(Directory):
    """
    Like directory, but all given paths must be lower-case,
    and will be resolved to the actual correct case.

    The one exception is the constructor argument:
    It _must_ be in the correct case.
    """

    def __init__(self, path, create_if_missing=False):
        if False:
            print('Hello World!')
        super().__init__(path, create_if_missing)
        self.cache = {(): ()}
        self.listings = {}

    def __repr__(self):
        if False:
            return 10
        return f"Directory({self.path.decode(errors='replace')})"

    def actual_name(self, stem: list, name: str) -> str:
        if False:
            return 10
        "\n        If the (lower-case) path that's given in stem exists,\n        fetches the actual name for the given lower-case name.\n        "
        try:
            listing = self.listings[tuple(stem)]
        except KeyError:
            try:
                filelist = os.listdir(os.path.join(self.path, *stem))
            except FileNotFoundError:
                filelist = []
            listing = {}
            for filename in filelist:
                if filename.lower() != filename:
                    listing[filename.lower()] = filename
            self.listings[tuple(stem)] = listing
        try:
            return listing[name]
        except KeyError:
            return name

    def resolve(self, parts) -> Union[str, bytes]:
        if False:
            i = 10
            return i + 15
        parts = [part.lower() for part in parts]
        i = 0
        for i in range(len(parts), -1, -1):
            try:
                result = list(self.cache[tuple(parts[:i])])
                break
            except KeyError:
                pass
        else:
            raise RuntimeError('code flow error')
        for part in parts[i:]:
            result.append(self.actual_name(result, part))
            self.cache[tuple(parts[:len(result)])] = tuple(result)
        return os.path.join(self.path, *result)

    def list(self, parts) -> typing.Generator[str | bytes, None, None]:
        if False:
            i = 10
            return i + 15
        for name in super().list(parts):
            yield name.lower()