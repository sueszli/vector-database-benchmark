from __future__ import annotations
import itertools
import os
import pathlib
import sys
import tempfile
from itertools import islice
from typing import Union
from file_read_backwards import FileReadBackwards

class Path(type(pathlib.Path())):

    @staticmethod
    def mkdtemp(*args, **kwargs) -> Path:
        if False:
            return 10
        return Path(tempfile.mkdtemp(*args, **kwargs))

    @staticmethod
    def fix_win_long_file(path: Path):
        if False:
            while True:
                i = 10
        '"\n        String representation of Path-like object with work around for Windows long filename issue.\n        '
        if sys.platform == 'win32':
            return '\\\\?\\' + str(path)
        return str(path)

    def normalize_to(self, base: str=None) -> Path:
        if False:
            return 10
        "Return a relative path if 'self' is relative to base.\n        Return an absolute path overwise.\n        "
        if base is None:
            return self
        try:
            return self.relative_to(Path(base))
        except ValueError:
            pass
        return self

    def size(self, include_dir_sizes: bool=True) -> int:
        if False:
            return 10
        ' Return the size of this file or directory (recursively).\n\n        Args:\n            include_dir_sizes: If True, return the size of files and directories, not the size of files only.\n\n        Returns: The size of this file or directory.\n        '
        if not self.exists():
            return 0
        if self.is_file():
            return self.stat().st_size
        size = os.path.getsize(self.absolute()) if include_dir_sizes else 0
        for (root, dir_names, file_names) in os.walk(self):
            names = itertools.chain(dir_names, file_names) if include_dir_sizes else file_names
            paths = (os.path.join(root, name) for name in names)
            for p in paths:
                try:
                    size += os.path.getsize(p)
                except OSError:
                    pass
        return size

    def startswith(self, text: str) -> bool:
        if False:
            return 10
        return self.match(f'{text}*')

    def endswith(self, text: str) -> bool:
        if False:
            while True:
                i = 10
        return self.match(f'*{text}')

class PosixPath(Path, pathlib.PurePosixPath):
    __slots__ = ()

class WindowsPath(Path, pathlib.PureWindowsPath):
    __slots__ = ()

def tail(file_name: Union[str, Path], count: int=1) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Tail a file and get `count` lines from the end'
    with FileReadBackwards(file_name) as f:
        lines = list(islice(f, count))
        return '\n'.join(reversed(lines))