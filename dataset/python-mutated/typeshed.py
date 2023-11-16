"""
Defines an abstract interface representing a typeshed that allows
us to use different "backings" (e.g. in-memory for tests or raw
data pulled from github vs a local directory where we have one)
in most of our patching code.
"""
import abc
import contextlib
import pathlib
import shutil
import tempfile
import zipfile
from typing import Dict, Iterable, Iterator, Mapping, Optional, Set

class Typeshed(abc.ABC):
    """
    Representation of a collection of Python stub files.
    """

    @abc.abstractclassmethod
    def get_file_content(self, path: pathlib.Path) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        Return content of the given path, or `None` if the content is not available.\n        Paths are all relative to typeshed root.\n        This method is allowed to return `None` even for paths that come from the\n        return value of `all_files()`. But such cases are expected to be rare.\n        '
        raise NotImplementedError()

    @abc.abstractclassmethod
    def all_files(self) -> Iterable[pathlib.Path]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return paths to all contained files (directory excluded).\n        Paths are all relative to typeshed root.\n        Elements in the returned iterable is not guaranteed to follow any particular\n        order.\n        '
        raise NotImplementedError()

def _write_to_directory(typeshed: Typeshed, root: pathlib.Path) -> None:
    if False:
        while True:
            i = 10
    for path in typeshed.all_files():
        content = typeshed.get_file_content(path)
        if content is None:
            continue
        full_path = root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

@contextlib.contextmanager
def _create_temporary_typeshed_directory(typeshed: Typeshed) -> Iterator[pathlib.Path]:
    if False:
        print('Hello World!')
    with tempfile.TemporaryDirectory() as temporary_root:
        temporary_root_path = pathlib.Path(temporary_root)
        _write_to_directory(typeshed, temporary_root_path)
        yield temporary_root_path

def write_to_directory(typeshed: Typeshed, target: pathlib.Path) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Write the given `Typeshed` into a directory rooted at `target` on the filesystem.\n\n    The `target` directory is assumed to be nonexistent before this function gets\n    invoked.\n    '
    if target.exists():
        raise ValueError(f'Cannot write to file that already exists: `{target}`')
    with _create_temporary_typeshed_directory(typeshed) as temporary_root:
        temporary_root.rename(target)

def write_to_zip(typeshed: Typeshed, target: pathlib.Path) -> None:
    if False:
        while True:
            i = 10
    '\n    Write the given `Typeshed` into a zip file `target`.\n\n    File at `target` path is assumed to be nonexistent before this function gets\n    invoked. The `target` path is also assumed to have `.zip` as its suffix.\n    '
    if target.exists():
        raise ValueError(f'Cannot write to file that already exists: `{target}`')
    if target.suffix != '.zip':
        raise ValueError(f'Cannot write to `{target}` as zip file: wrong suffix.')
    with _create_temporary_typeshed_directory(typeshed) as temporary_root:
        shutil.make_archive(str(target.with_suffix('')), format='zip', root_dir=temporary_root)

class MemoryBackedTypeshed(Typeshed):
    """
    A typeshed backed up by in-memory content. Essentially a wrapper around
    a dictonary from paths to their contents.
    This class is mostly useful for testing.
    """
    contents: Mapping[pathlib.Path, str]

    def __init__(self, contents: Mapping[pathlib.Path, str]) -> None:
        if False:
            return 10
        self.contents = contents

    def all_files(self) -> Iterable[pathlib.Path]:
        if False:
            i = 10
            return i + 15
        return self.contents.keys()

    def get_file_content(self, path: pathlib.Path) -> Optional[str]:
        if False:
            print('Hello World!')
        return self.contents.get(path, None)

class DirectoryBackedTypeshed(Typeshed):
    """
    A typeshed backed up by a directory that lives on the filesystem.

    For simplicity, we assume that files in this directory remain unchanged. If
    the assumption does not hold, e.g. when files are added/removed/changed after
    the creation of a `DirectoryBacked` object, the behaviors of its methods
    become undefined.
    """
    root: pathlib.Path
    files: Set[pathlib.Path]

    def __init__(self, root: pathlib.Path) -> None:
        if False:
            return 10
        self.root = root
        self.files = {path.relative_to(root) for path in root.rglob('*') if path.is_file()}

    def all_files(self) -> Iterable[pathlib.Path]:
        if False:
            return 10
        return self.files

    def get_file_content(self, path: pathlib.Path) -> Optional[str]:
        if False:
            print('Hello World!')
        return (self.root / path).read_text() if path in self.files else None

class ZipBackedTypeshed(Typeshed):
    """
    A typeshed backed up by a zipball that lives on the filesystem.

    For simplicity, we assume that this zipfile remains unchanged. If the assumption
    does not hold, e.g. when this file gets added/removed/changed after the creation of
    the corresponding `ZipBackedTypeshed` object, the behaviors of its methods become
    undefined.
    """
    zip_file: zipfile.ZipFile

    def __init__(self, zip_file_path: pathlib.Path) -> None:
        if False:
            i = 10
            return i + 15
        self.zip_file = zipfile.ZipFile(zip_file_path)

    def all_files(self) -> Iterable[pathlib.Path]:
        if False:
            print('Hello World!')
        return [pathlib.Path(zip_info.filename) for zip_info in self.zip_file.infolist() if not zip_info.is_dir()]

    def get_file_content(self, path: pathlib.Path) -> Optional[str]:
        if False:
            while True:
                i = 10
        try:
            return self.zip_file.read(str(path)).decode('utf-8')
        except (KeyError, ValueError):
            return None

class PatchedTypeshed(Typeshed):
    """
    A typeshed backed up by another `Typeshed` object and a set of patch results
    that overwrite file contents in the base `Typeshed` object.

    Patches are specified as a dictionary from paths to either a `str` or `None`.
    When the value is a string, it serves as the new content for the corresponding
    file (or the content of a new file if the file did not exist before). When the
    value is `None`, it indicates that the corresponding file will be removed.
    """
    base: Typeshed
    updated_files: Dict[pathlib.Path, str]
    removed_files: Set[pathlib.Path]

    def __init__(self, base: Typeshed, patch_results: Mapping[pathlib.Path, Optional[str]]) -> None:
        if False:
            return 10
        self.base = base
        self.updated_files = {path: content for (path, content) in patch_results.items() if content is not None}
        self.removed_files = {path for (path, content) in patch_results.items() if content is None}

    def all_files(self) -> Iterable[pathlib.Path]:
        if False:
            i = 10
            return i + 15
        return (set(self.base.all_files()) | self.updated_files.keys()) - self.removed_files

    def get_file_content(self, path: pathlib.Path) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        if path in self.removed_files:
            return None
        updated_content = self.updated_files.get(path, None)
        if updated_content is not None:
            return updated_content
        return self.base.get_file_content(path)