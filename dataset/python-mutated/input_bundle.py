import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Iterator, Optional
from vyper.exceptions import JSONError
PathLike = Path | PurePath

@dataclass
class CompilerInput:
    source_id: int
    path: PathLike

    @staticmethod
    def from_string(source_id: int, path: PathLike, file_contents: str) -> 'CompilerInput':
        if False:
            print('Hello World!')
        try:
            s = json.loads(file_contents)
            return ABIInput(source_id, path, s)
        except (ValueError, TypeError):
            return FileInput(source_id, path, file_contents)

@dataclass
class FileInput(CompilerInput):
    source_code: str

@dataclass
class ABIInput(CompilerInput):
    abi: Any

class _NotFound(Exception):
    pass

def _normpath(path):
    if False:
        print('Hello World!')
    return path.__class__(os.path.normpath(path))

class InputBundle:
    search_paths: list[PathLike]

    def __init__(self, search_paths):
        if False:
            i = 10
            return i + 15
        self.search_paths = search_paths
        self._source_id_counter = 0
        self._source_ids: dict[PathLike, int] = {}

    def _load_from_path(self, path):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(f'not implemented! {self.__class__}._load_from_path()')

    def _generate_source_id(self, path: PathLike) -> int:
        if False:
            print('Hello World!')
        if path not in self._source_ids:
            self._source_ids[path] = self._source_id_counter
            self._source_id_counter += 1
        return self._source_ids[path]

    def load_file(self, path: PathLike | str) -> CompilerInput:
        if False:
            print('Hello World!')
        tried = []
        for sp in reversed(self.search_paths):
            to_try = sp / path
            to_try = _normpath(to_try)
            try:
                res = self._load_from_path(to_try)
                break
            except _NotFound:
                tried.append(to_try)
        else:
            formatted_search_paths = '\n'.join(['  ' + str(p) for p in tried])
            raise FileNotFoundError(f'could not find {path} in any of the following locations:\n{formatted_search_paths}')
        if isinstance(res, FileInput):
            return CompilerInput.from_string(res.source_id, res.path, res.source_code)
        return res

    def add_search_path(self, path: PathLike) -> None:
        if False:
            while True:
                i = 10
        self.search_paths.append(path)

    @contextlib.contextmanager
    def search_path(self, path: Optional[PathLike]) -> Iterator[None]:
        if False:
            while True:
                i = 10
        if path is None:
            yield
        else:
            self.search_paths.append(path)
            try:
                yield
            finally:
                self.search_paths.pop()

class FilesystemInputBundle(InputBundle):

    def _load_from_path(self, path: Path) -> CompilerInput:
        if False:
            for i in range(10):
                print('nop')
        try:
            with path.open() as f:
                code = f.read()
        except FileNotFoundError:
            raise _NotFound(path)
        source_id = super()._generate_source_id(path)
        return FileInput(source_id, path, code)

class JSONInputBundle(InputBundle):
    input_json: dict[PurePath, Any]

    def __init__(self, input_json, search_paths):
        if False:
            print('Hello World!')
        super().__init__(search_paths)
        self.input_json = {}
        for (path, item) in input_json.items():
            path = _normpath(path)
            assert path not in self.input_json
            self.input_json[_normpath(path)] = item

    def _load_from_path(self, path: PurePath) -> CompilerInput:
        if False:
            while True:
                i = 10
        try:
            value = self.input_json[path]
        except KeyError:
            raise _NotFound(path)
        source_id = super()._generate_source_id(path)
        if 'content' in value:
            return FileInput(source_id, path, value['content'])
        if 'abi' in value:
            return ABIInput(source_id, path, value['abi'])
        raise JSONError(f"Unexpected type in file: '{path}'")