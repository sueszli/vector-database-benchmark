"""Watch parts of the file system for changes."""
from __future__ import annotations
from typing import AbstractSet, Iterable, NamedTuple
from mypy.fscache import FileSystemCache

class FileData(NamedTuple):
    st_mtime: float
    st_size: int
    hash: str

class FileSystemWatcher:
    """Watcher for file system changes among specific paths.

    All file system access is performed using FileSystemCache. We
    detect changed files by stat()ing them all and comparing hashes
    of potentially changed files. If a file has both size and mtime
    unmodified, the file is assumed to be unchanged.

    An important goal of this class is to make it easier to eventually
    use file system events to detect file changes.

    Note: This class doesn't flush the file system cache. If you don't
    manually flush it, changes won't be seen.
    """

    def __init__(self, fs: FileSystemCache) -> None:
        if False:
            i = 10
            return i + 15
        self.fs = fs
        self._paths: set[str] = set()
        self._file_data: dict[str, FileData | None] = {}

    def dump_file_data(self) -> dict[str, tuple[float, int, str]]:
        if False:
            for i in range(10):
                print('nop')
        return {k: v for (k, v) in self._file_data.items() if v is not None}

    def set_file_data(self, path: str, data: FileData) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._file_data[path] = data

    def add_watched_paths(self, paths: Iterable[str]) -> None:
        if False:
            return 10
        for path in paths:
            if path not in self._paths:
                self._file_data[path] = None
        self._paths |= set(paths)

    def remove_watched_paths(self, paths: Iterable[str]) -> None:
        if False:
            i = 10
            return i + 15
        for path in paths:
            if path in self._file_data:
                del self._file_data[path]
        self._paths -= set(paths)

    def _update(self, path: str) -> None:
        if False:
            while True:
                i = 10
        st = self.fs.stat(path)
        hash_digest = self.fs.hash_digest(path)
        self._file_data[path] = FileData(st.st_mtime, st.st_size, hash_digest)

    def _find_changed(self, paths: Iterable[str]) -> AbstractSet[str]:
        if False:
            print('Hello World!')
        changed = set()
        for path in paths:
            old = self._file_data[path]
            try:
                st = self.fs.stat(path)
            except FileNotFoundError:
                if old is not None:
                    changed.add(path)
                    self._file_data[path] = None
            else:
                if old is None:
                    changed.add(path)
                    self._update(path)
                elif st.st_size != old.st_size or int(st.st_mtime) != int(old.st_mtime):
                    new_hash = self.fs.hash_digest(path)
                    self._update(path)
                    if st.st_size != old.st_size or new_hash != old.hash:
                        changed.add(path)
        return changed

    def find_changed(self) -> AbstractSet[str]:
        if False:
            print('Hello World!')
        'Return paths that have changes since the last call, in the watched set.'
        return self._find_changed(self._paths)

    def update_changed(self, remove: list[str], update: list[str]) -> AbstractSet[str]:
        if False:
            return 10
        'Alternative to find_changed() given explicit changes.\n\n        This only calls self.fs.stat() on added or updated files, not\n        on all files.  It believes all other files are unchanged!\n\n        Implies add_watched_paths() for add and update, and\n        remove_watched_paths() for remove.\n        '
        self.remove_watched_paths(remove)
        self.add_watched_paths(update)
        return self._find_changed(update)