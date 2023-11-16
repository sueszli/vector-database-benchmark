import fnmatch
import os.path
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from robot.errors import DataError
from robot.output import LOGGER
from robot.utils import get_error_message

class SuiteStructure(ABC):
    source: 'Path|None'
    init_file: 'Path|None'
    children: 'list[SuiteStructure]|None'

    def __init__(self, extensions: 'ValidExtensions', source: 'Path|None', init_file: 'Path|None'=None, children: 'Sequence[SuiteStructure]|None'=None):
        if False:
            i = 10
            return i + 15
        self._extensions = extensions
        self.source = source
        self.init_file = init_file
        self.children = list(children) if children is not None else None

    @property
    def extension(self) -> 'str|None':
        if False:
            while True:
                i = 10
        source = self._get_source_file()
        return self._extensions.get_extension(source) if source else None

    @abstractmethod
    def _get_source_file(self) -> 'Path|None':
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def visit(self, visitor: 'SuiteStructureVisitor'):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class SuiteFile(SuiteStructure):
    source: Path

    def __init__(self, extensions: 'ValidExtensions', source: Path):
        if False:
            print('Hello World!')
        super().__init__(extensions, source)

    def _get_source_file(self) -> Path:
        if False:
            print('Hello World!')
        return self.source

    def visit(self, visitor: 'SuiteStructureVisitor'):
        if False:
            for i in range(10):
                print('nop')
        visitor.visit_file(self)

class SuiteDirectory(SuiteStructure):
    children: 'list[SuiteStructure]'

    def __init__(self, extensions: 'ValidExtensions', source: 'Path|None'=None, init_file: 'Path|None'=None, children: Sequence[SuiteStructure]=()):
        if False:
            return 10
        super().__init__(extensions, source, init_file, children)

    def _get_source_file(self) -> 'Path|None':
        if False:
            return 10
        return self.init_file

    @property
    def is_multi_source(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.source is None

    def add(self, child: 'SuiteStructure'):
        if False:
            i = 10
            return i + 15
        self.children.append(child)

    def visit(self, visitor: 'SuiteStructureVisitor'):
        if False:
            while True:
                i = 10
        visitor.visit_directory(self)

class SuiteStructureVisitor:

    def visit_file(self, structure: SuiteFile):
        if False:
            return 10
        pass

    def visit_directory(self, structure: SuiteDirectory):
        if False:
            return 10
        self.start_directory(structure)
        for child in structure.children:
            child.visit(self)
        self.end_directory(structure)

    def start_directory(self, structure: SuiteDirectory):
        if False:
            i = 10
            return i + 15
        pass

    def end_directory(self, structure: SuiteDirectory):
        if False:
            i = 10
            return i + 15
        pass

class SuiteStructureBuilder:
    ignored_prefixes = ('_', '.')
    ignored_dirs = ('CVS',)

    def __init__(self, extensions: Sequence[str]=('.robot', '.rbt', '.robot.rst'), included_files: Sequence[str]=()):
        if False:
            for i in range(10):
                print('nop')
        self.extensions = ValidExtensions(extensions, included_files)
        self.included_files = IncludedFiles(included_files)

    def build(self, *paths: Path) -> SuiteStructure:
        if False:
            return 10
        if len(paths) == 1:
            return self._build(paths[0])
        return self._build_multi_source(paths)

    def _build(self, path: Path) -> SuiteStructure:
        if False:
            while True:
                i = 10
        if path.is_file():
            return SuiteFile(self.extensions, path)
        return self._build_directory(path)

    def _build_directory(self, path: Path) -> SuiteStructure:
        if False:
            for i in range(10):
                print('nop')
        structure = SuiteDirectory(self.extensions, path)
        for item in self._list_dir(path):
            if self._is_init_file(item):
                if structure.init_file:
                    LOGGER.error(f"Ignoring second test suite init file '{item}'.")
                else:
                    structure.init_file = item
            elif self._is_included(item):
                structure.add(self._build(item))
            else:
                LOGGER.info(f"Ignoring file or directory '{item}'.")
        return structure

    def _list_dir(self, path: Path) -> 'list[Path]':
        if False:
            for i in range(10):
                print('nop')
        try:
            return sorted(path.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            raise DataError(f"Reading directory '{path}' failed: {get_error_message()}")

    def _is_init_file(self, path: Path) -> bool:
        if False:
            print('Hello World!')
        return path.stem.lower() == '__init__' and self.extensions.match(path) and path.is_file()

    def _is_included(self, path: Path) -> bool:
        if False:
            while True:
                i = 10
        if path.name.startswith(self.ignored_prefixes):
            return False
        if path.is_dir():
            return path.name not in self.ignored_dirs
        if not path.is_file():
            return False
        if not self.extensions.match(path):
            return False
        return self.included_files.match(path)

    def _build_multi_source(self, paths: Iterable[Path]) -> SuiteStructure:
        if False:
            i = 10
            return i + 15
        structure = SuiteDirectory(self.extensions)
        for path in paths:
            if self._is_init_file(path):
                if structure.init_file:
                    raise DataError('Multiple init files not allowed.')
                structure.init_file = path
            else:
                structure.add(self._build(path))
        return structure

class ValidExtensions:

    def __init__(self, extensions: Sequence[str], included_files: Sequence[str]=()):
        if False:
            while True:
                i = 10
        self.extensions = {ext.lstrip('.').lower() for ext in extensions}
        for pattern in included_files:
            ext = os.path.splitext(pattern)[1]
            if ext:
                self.extensions.add(ext.lstrip('.').lower())

    def match(self, path: Path) -> bool:
        if False:
            while True:
                i = 10
        for ext in self._extensions_from(path):
            if ext in self.extensions:
                return True
        return False

    def get_extension(self, path: Path) -> str:
        if False:
            while True:
                i = 10
        for ext in self._extensions_from(path):
            if ext in self.extensions:
                return ext
        return path.suffix.lower()[1:]

    def _extensions_from(self, path: Path) -> Iterator[str]:
        if False:
            return 10
        suffixes = path.suffixes
        while suffixes:
            yield ''.join(suffixes).lower()[1:]
            suffixes.pop(0)

class IncludedFiles:

    def __init__(self, patterns: 'Sequence[str|Path]'=()):
        if False:
            for i in range(10):
                print('nop')
        self.patterns = [self._compile(i) for i in patterns]

    def _compile(self, pattern: 'str|Path') -> 're.Pattern':
        if False:
            i = 10
            return i + 15
        pattern = self._dir_to_recursive(self._path_to_abs(self._normalize(pattern)))
        parts = [self._translate(p) for p in pattern.split('**')]
        return re.compile('.*'.join(parts), re.IGNORECASE)

    def _normalize(self, pattern: 'str|Path') -> str:
        if False:
            print('Hello World!')
        if isinstance(pattern, Path):
            pattern = str(pattern)
        return os.path.normpath(pattern).replace('\\', '/')

    def _path_to_abs(self, pattern: str) -> str:
        if False:
            i = 10
            return i + 15
        if '/' in pattern or '.' not in pattern or os.path.exists(pattern):
            pattern = os.path.abspath(pattern).replace('\\', '/')
        return pattern

    def _dir_to_recursive(self, pattern: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        if '.' not in os.path.basename(pattern) or os.path.isdir(pattern):
            pattern += '/**'
        return pattern

    def _translate(self, glob_pattern: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        re_pattern = fnmatch.translate(glob_pattern)[4:-3]
        return re_pattern.replace('.*', '[^/]*')

    def match(self, path: Path) -> bool:
        if False:
            i = 10
            return i + 15
        if not self.patterns:
            return True
        return self._match(path.name) or self._match(str(path))

    def _match(self, path: str) -> bool:
        if False:
            return 10
        path = self._normalize(path)
        return any((p.fullmatch(path) for p in self.patterns))