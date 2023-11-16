from __future__ import annotations
import os
import re
from bisect import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Sequence, cast, ItemsView, Dict

class TorrentFileTree:
    """
    A tree of directories that contain other directories and files.
    """

    @dataclass
    class Directory:
        """
        A directory that contains other directories and files.
        """
        directories: defaultdict[str, TorrentFileTree.Directory] = field(default_factory=dict)
        files: list[TorrentFileTree.File] = field(default_factory=list)
        collapsed: bool = True
        size: int = 0

        def calc_size(self):
            if False:
                return 10
            '\n            Calculate the size of this Directory, assuming all subdirectories already have their size calculated.\n            '
            self.size = sum((dir.size for dir in self.directories.values())) + sum((f.size for f in self.files))

        def iter_dirs(self) -> Generator[TorrentFileTree.Directory, None, None]:
            if False:
                while True:
                    i = 10
            '\n            Iterate through the subdirectories in this directory and then this directory itself.\n\n            We do it this way so that calc_size() can be easily/efficiently executed!\n            '
            for directory in self.directories.values():
                for entry in directory.iter_dirs():
                    yield entry
            yield self

        def tostr(self, depth: int=0, name: str='') -> str:
            if False:
                print('Hello World!')
            '\n            Create a beautifully formatted string representation of this directory.\n            '
            tab = '\t'
            if self.collapsed:
                return '\n' + '\t' * depth + f'CollapsedDirectory({name!r}, {self.size} bytes)'
            has_no_directories = len(self.directories) == 0
            pretty_directories = ','.join((v.tostr(depth + 2, k) for (k, v) in self.directories.items()))
            dir_closure = '' if has_no_directories else '\n' + tab * (depth + 1)
            pretty_directories = f'\n{tab * (depth + 1)}directories=[{pretty_directories}{dir_closure}]'
            pretty_files = ''.join(('\n' + v.tostr(depth + 2) for v in self.files))
            pretty_files = f'\n{tab * (depth + 1)}files=[{pretty_files}]'
            return '\n' + '\t' * depth + f'Directory({name!r},{pretty_directories},{pretty_files}, {self.size} bytes)'

    @dataclass(unsafe_hash=True)
    class File:
        """
        A File object that has a name (relative to its parent directory) and a file index in the torrent's file list.
        """
        name: str
        index: int
        size: int = 0
        _sort_pattern = re.compile('([0-9]+)')

        def tostr(self, depth: int=0) -> str:
            if False:
                return 10
            '\n            Create a beautifully formatted string representation of this File.\n            '
            return '\t' * depth + f'File({self.index}, {self.name}, {self.size} bytes)'

        def sort_key(self) -> Sequence[int | str]:
            if False:
                i = 10
                return i + 15
            '\n            Sort File instances using natural sort based on their names, which SHOULD be unique.\n            '
            return tuple((int(part) if part.isdigit() else part for part in self._sort_pattern.split(self.name)))

        def __lt__(self, other) -> bool:
            if False:
                while True:
                    i = 10
            '\n            Python 3.8 quirk/shortcoming is that File needs to be a SupportsRichComparisonT (instead of using a key).\n            '
            return self.sort_key() < other.sort_key()

        def __le__(self, other) -> bool:
            if False:
                return 10
            '\n            Python 3.8 quirk/shortcoming is that File needs to be a SupportsRichComparisonT (instead of using a key).\n            '
            return self.sort_key() <= other.sort_key()

        def __gt__(self, other) -> bool:
            if False:
                while True:
                    i = 10
            '\n            Python 3.8 quirk/shortcoming is that File needs to be a SupportsRichComparisonT (instead of using a key).\n            '
            return self.sort_key() > other.sort_key()

        def __ge__(self, other) -> bool:
            if False:
                print('Hello World!')
            '\n            Python 3.8 quirk/shortcoming is that File needs to be a SupportsRichComparisonT (instead of using a key).\n            '
            return self.sort_key() >= other.sort_key()

        def __eq__(self, other) -> bool:
            if False:
                i = 10
                return i + 15
            '\n            Python 3.8 quirk/shortcoming is that File needs to be a SupportsRichComparisonT (instead of using a key).\n            '
            return self.sort_key() == other.sort_key()

        def __ne__(self, other) -> bool:
            if False:
                print('Hello World!')
            '\n            Python 3.8 quirk/shortcoming is that File needs to be a SupportsRichComparisonT (instead of using a key).\n            '
            return self.sort_key() != other.sort_key()

    def __init__(self, file_storage) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct an empty tree data structure belonging to the given file storage.\n\n        Note that the file storage contents are not loaded in yet at this point.\n        '
        self.root = TorrentFileTree.Directory()
        self.root.collapsed = False
        self.file_storage = file_storage
        self.paths: Dict[Path, TorrentFileTree.Directory | TorrentFileTree.File] = {}

    def __str__(self) -> str:
        if False:
            return 10
        '\n        Represent the tree as a string, which is actually just the tostr() of its root directory.\n        '
        return f'TorrentFileTree({self.root.tostr()}\n)'

    @classmethod
    def from_lt_file_storage(cls, file_storage):
        if False:
            while True:
                i = 10
        '\n        Load in the tree contents from the given file storage, sorting the files in each directory.\n        '
        tree = cls(file_storage)
        for i in range(file_storage.num_files()):
            full_file_path = Path(file_storage.file_path(i))
            (*subdirs, fname) = full_file_path.parts
            current_dir = tree.root
            full_path = Path('')
            for subdir in subdirs:
                d = current_dir.directories.get(subdir, TorrentFileTree.Directory())
                current_dir.directories[subdir] = d
                current_dir = d
                full_path = full_path / subdir
                tree.paths[full_path] = d
            file_instance = cls.File(fname, i, file_storage.file_size(i))
            current_dir.files.append(file_instance)
            tree.paths[full_file_path] = file_instance
        for directory in tree.root.iter_dirs():
            directory.files.sort()
            directory.calc_size()
        return tree

    def expand(self, path: Path) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Expand all directories that are necessary to view the given path.\n        '
        current_dir = self.root
        for directory in path.parts:
            if directory not in current_dir.directories:
                break
            current_dir = current_dir.directories[directory]
            current_dir.collapsed = False

    def collapse(self, path: Path) -> None:
        if False:
            while True:
                i = 10
        '\n        Collapse ONLY the specific given directory.\n        '
        element = self.find(path)
        if isinstance(element, TorrentFileTree.Directory) and element != self.root:
            element.collapsed = True

    def find(self, path: Path) -> Directory | File | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the Directory or File object at the given path, or None if it does not exist.\n\n        Searching for files is "expensive" (use libtorrent instead).\n        '
        if path == Path(''):
            return self.root
        current_dir = self.root
        for directory in path.parts:
            if directory not in current_dir.directories:
                if len(current_dir.files) == 0:
                    return None
                search = self.File(directory, 0)
                found_at = bisect(current_dir.files, search)
                element = current_dir.files[found_at - 1]
                return element if element == search else None
            current_dir = current_dir.directories[directory]
        return current_dir

    def path_is_dir(self, path: Path) -> bool:
        if False:
            while True:
                i = 10
        '\n        Check if the given path points to a Directory (instead of a File).\n        '
        if path == Path(''):
            return True
        current_dir = self.root
        for directory in path.parts:
            if directory not in current_dir.directories:
                return False
            current_dir = current_dir.directories[directory]
        return True

    def find_next_directory(self, from_path: Path) -> tuple[Directory, Path] | None:
        if False:
            i = 10
            return i + 15
        '\n        Get the next unvisited directory from a given path.\n\n        When we ran out of files, we have to go up in the tree. However, when we go up, we may immediately be at the\n        end of the list of that parent directory and we may have to go up again. If we are at the end of the list all\n        the way up to the root of the tree, we return None.\n        '
        from_parts = from_path.parts
        for i in range(1, len(from_parts) + 1):
            parent_path = Path(os.sep.join(from_parts[:-i]))
            parent = self.find(parent_path)
            dir_in_parent = from_parts[-i]
            dir_indices = list(parent.directories.keys())
            index_in_parent = dir_indices.index(dir_in_parent)
            if index_in_parent != len(dir_indices) - 1:
                dirname = dir_indices[index_in_parent + 1]
                return (parent.directories[dirname], parent_path / dirname)
            if len(parent.files) > 0:
                return (parent, parent_path / parent.files[0].name)
        return None

    def _view_get_fetch_path_and_dir(self, start_path: tuple[Directory, Path] | Path) -> tuple[Directory, Path, Path]:
        if False:
            return 10
        "\n        Given a start path, which may be a file, get the containing Directory object and directory path.\n\n        In the case that we start from a given Directory object and a file path, we only correct the file path to\n        start at the given Directory's path.\n        "
        if isinstance(start_path, Path):
            fetch_path = start_path if self.path_is_dir(start_path) else start_path.parent
            fetch_directory = cast(TorrentFileTree.Directory, self.find(fetch_path))
            return (fetch_directory, fetch_path, start_path)
        (fetch_directory, fetch_path) = start_path
        requested_fetch_path = fetch_path
        if not self.path_is_dir(fetch_path):
            fetch_path = fetch_path.parent
        return (fetch_directory, fetch_path, requested_fetch_path)

    def _view_up_after_files(self, number: int, fetch_path: Path) -> list[str]:
        if False:
            i = 10
            return i + 15
        '\n        Run up the tree to the next available directory (if it exists) and continue building a view.\n        '
        next_dir_desc = self.find_next_directory(fetch_path)
        view = []
        if next_dir_desc is None:
            return view
        (next_dir, next_dir_path) = next_dir_desc
        view.append(str(next_dir_path))
        number -= 1
        if number == 0:
            return view
        return view + self.view((next_dir, next_dir_path), number)

    def _view_process_directories(self, number: int, directory_items: ItemsView[str, Directory], fetch_path: Path) -> tuple[list[str], int]:
        if False:
            while True:
                i = 10
        '\n        Process the directories dictionary of a given (parent directory) path.\n\n        Note that we only need to process the first directory and the remainder is visited through recursion.\n        '
        view = []
        try:
            (dirname, dirobj) = next(iter(directory_items))
        except StopIteration:
            return (view, number)
        full_path = fetch_path / dirname
        view.append(str(full_path))
        number -= 1
        if number == 0:
            return (view, number)
        if not dirobj.collapsed:
            elems = self.view((dirobj, full_path), number)
            view += elems
            number -= len(elems)
        return (view, number)

    def view(self, start_path: tuple[Directory, Path] | Path, number: int) -> list[str]:
        if False:
            i = 10
            return i + 15
        '\n        Construct a view of a given number of path names (directories and files) in the tree.\n\n        The view is constructed AFTER the given starting path. To view the root folder contents, simply call this\n        method with Path("") or Path(".").\n        '
        (fetch_directory, fetch_path, element_path) = self._view_get_fetch_path_and_dir(start_path)
        if fetch_directory.collapsed:
            return []
        view = []
        if self.path_is_dir(element_path):
            (view, number) = self._view_process_directories(number, fetch_directory.directories.items(), fetch_path)
            if number == 0:
                return view
            if len(view) > 0 and len(fetch_directory.files) > 0 and (view[-1] == self.file_storage.file_path(fetch_directory.files[0].index)):
                files = [str(element_path / f.name) for f in fetch_directory.files[1:number + 1]]
            else:
                files = [str(element_path / f.name) for f in fetch_directory.files[:number]]
        else:
            fetch_index = bisect(fetch_directory.files, self.File(element_path.parts[-1], 0))
            files = [str(fetch_path / f.name) for f in fetch_directory.files[fetch_index:fetch_index + number]]
        view += files
        number -= len(files)
        return view if number == 0 else view + self._view_up_after_files(number, fetch_path)