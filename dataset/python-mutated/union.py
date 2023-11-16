"""
Provides Union, a utility class for combining multiple FSLikeObjects to a
single one.
"""
from io import UnsupportedOperation
from .abstract import FSLikeObject
from .path import Path

class Union(FSLikeObject):
    """
    FSLikeObject that provides a structure for mounting several path objects.

    Unlike in POSIX, mounts may overlap.
    If multiple mounts match for a directory, those that have a higher
    priority are preferred.
    In case of equal priorities, later mounts are preferred.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.mounts = []
        self.dirstructure = {}

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        content = ', '.join([f'{repr(pnt[1])} @ {repr(pnt[0])}' for pnt in self.mounts])
        return f'Union({content})'

    @property
    def root(self):
        if False:
            return 10
        return UnionPath(self, [])

    def add_mount(self, pathobj: Path, mountpoint, priority: int) -> None:
        if False:
            print('Hello World!')
        '\n        This method should not be called directly; instead, use the mount\n        method of Path objects that were obtained from this.\n\n        Mounts pathobj at mountpoint, with the given priority.\n        '
        if not isinstance(pathobj, Path):
            raise PermissionError(f'only a fslike.Path can be mounted, not {type(pathobj)}')
        idx = len(self.mounts) - 1
        while idx >= 0 and priority >= self.mounts[idx][2]:
            idx -= 1
        self.mounts.insert(idx + 1, (tuple(mountpoint), pathobj, priority))
        dirstructure = self.dirstructure
        for subdir in mountpoint:
            dirstructure = dirstructure.setdefault(subdir, {})

    def remove_mount(self, search_mountpoint, source_pathobj: Path=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove a mount from the union by searching for the source\n        that provides the given mountpoint.\n        Additionally, can check if the source equals the given pathobj.\n        '
        unmount = []
        for (idx, (mountpoint, pathobj, _)) in enumerate(self.mounts):
            if mountpoint == tuple(search_mountpoint[:len(mountpoint)]):
                if not source_pathobj or source_pathobj == pathobj:
                    unmount.append(idx)
        if unmount:
            for idx in reversed(sorted(unmount)):
                del self.mounts[idx]
        else:
            raise ValueError('could not find mounted source')

    def candidate_paths(self, parts):
        if False:
            while True:
                i = 10
        '\n        Helper method.\n\n        Yields path objects from all mounts that match parts, in the order of\n        their priorities.\n        '
        for (mountpoint, pathobj, _) in self.mounts:
            cut_parts = tuple(parts[:len(mountpoint)])
            if mountpoint == cut_parts:
                yield pathobj.joinpath(parts[len(mountpoint):])

    def open_r(self, parts):
        if False:
            i = 10
            return i + 15
        for path in self.candidate_paths(parts):
            if path.is_file():
                return path.open_r()
        raise FileNotFoundError(b'/'.join(parts))

    def open_w(self, parts):
        if False:
            for i in range(10):
                print('nop')
        for path in self.candidate_paths(parts):
            if path.writable():
                return path.open_w()
        raise UnsupportedOperation('not writable: ' + b'/'.join(parts).decode(errors='replace'))

    def open_a(self, parts):
        if False:
            for i in range(10):
                print('nop')
        for path in self.candidate_paths(parts):
            if path.writable():
                return path.open_a()
        raise UnsupportedOperation('not appendable: ' + b'/'.join(parts).decode(errors='replace'))

    def resolve_r(self, parts):
        if False:
            i = 10
            return i + 15
        for path in self.candidate_paths(parts):
            if path.is_file() or path.is_dir():
                return path._resolve_r()
        return None

    def resolve_w(self, parts):
        if False:
            for i in range(10):
                print('nop')
        for path in self.candidate_paths(parts):
            if path.writable():
                return path._resolve_w()
        return None

    def list(self, parts):
        if False:
            print('Hello World!')
        duplicates = set()
        dir_exists = False
        dirstructure = self.dirstructure
        try:
            for subdir in parts:
                dirstructure = dirstructure[subdir]
            dir_exists = True
            yield from dirstructure
            duplicates.update(dirstructure)
        except KeyError:
            dir_exists = False
        for path in self.candidate_paths(parts):
            if path.is_file():
                raise NotADirectoryError(repr(path))
            if not path.is_dir():
                continue
            dir_exists = True
            for name in path.list():
                if name not in duplicates:
                    yield name
                    duplicates.add(name)
        if not dir_exists:
            raise FileNotFoundError(b'/'.join(parts))

    def filesize(self, parts) -> int:
        if False:
            return 10
        for path in self.candidate_paths(parts):
            if path.is_file():
                return path.filesize
        raise FileNotFoundError(b'/'.join(parts))

    def mtime(self, parts) -> float:
        if False:
            i = 10
            return i + 15
        for path in self.candidate_paths(parts):
            if path.exists():
                return path.mtime
        raise FileNotFoundError(b'/'.join(parts))

    def mkdirs(self, parts) -> None:
        if False:
            while True:
                i = 10
        for path in self.candidate_paths(parts):
            if path.writable():
                return path.mkdirs()
        return None

    def rmdir(self, parts) -> None:
        if False:
            return 10
        found = False
        for path in self.candidate_paths(parts):
            if path.is_dir():
                path.rmdir()
                found = True
        if not found:
            raise FileNotFoundError(b'/'.join(parts))

    def unlink(self, parts) -> None:
        if False:
            return 10
        found = False
        for path in self.candidate_paths(parts):
            if path.is_file():
                path.unlink()
                found = True
        if not found:
            raise FileNotFoundError(b'/'.join(parts))

    def touch(self, parts) -> None:
        if False:
            while True:
                i = 10
        for path in self.candidate_paths(parts):
            if path.writable():
                return path.touch()
        raise FileNotFoundError(b'/'.join(parts))

    def rename(self, srcparts, tgtparts) -> None:
        if False:
            return 10
        found = False
        for srcpath in self.candidate_paths(srcparts):
            if srcpath.exists():
                found = True
                if srcpath.writable():
                    for tgtpath in self.candidate_paths(tgtparts):
                        if tgtpath.writable():
                            return srcpath.rename(tgtpath)
        if found:
            raise UnsupportedOperation('read-only rename: ' + b'/'.join(srcparts).decode(errors='replace') + ' to ' + b'/'.join(tgtparts).decode(errors='replace'))
        raise FileNotFoundError(b'/'.join(srcparts))

    def is_file(self, parts) -> bool:
        if False:
            for i in range(10):
                print('nop')
        for path in self.candidate_paths(parts):
            if path.is_file():
                return True
        return False

    def is_dir(self, parts) -> bool:
        if False:
            return 10
        try:
            dirstructure = self.dirstructure
            for part in parts:
                dirstructure = dirstructure[part]
            return True
        except KeyError:
            pass
        for path in self.candidate_paths(parts):
            if path.is_dir():
                return True
        return False

    def writable(self, parts) -> bool:
        if False:
            return 10
        for path in self.candidate_paths(parts):
            if path.writable():
                return True
        return False

    def watch(self, parts, callback) -> bool:
        if False:
            for i in range(10):
                print('nop')
        watching = False
        for path in self.candidate_paths(parts):
            if path.exists():
                watching = watching or path.watch(callback)
        return watching

    def poll_watches(self):
        if False:
            return 10
        for (_, pathobj, _) in self.mounts:
            pathobj.poll_fs_watches()

class UnionPath(Path):
    """
    Provides an additional method for mounting an other path at this path.
    """

    def mount(self, pathobj: Path, priority: int=0) -> None:
        if False:
            print('Hello World!')
        "\n        Mounts pathobj here. All parent directories are 'created', if needed.\n        "
        return self.fsobj.add_mount(pathobj, self.parts, priority)

    def unmount(self, pathobj: Path=None) -> None:
        if False:
            print('Hello World!')
        '\n        Unmount a path from the union described by this path.\n        This is like "unmounting /home", no matter what the source was.\n        If you provide `pathobj`, that source is checked, additionally.\n\n        It will error if that path was not mounted.\n        '
        self.fsobj.remove_mount(self.parts, pathobj)