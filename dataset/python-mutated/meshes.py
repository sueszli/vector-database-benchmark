from pathlib import Path
from typing import Iterator
from ..diagnostic import Diagnostic
from .linter import Linter
MAX_MESH_FILE_SIZE = 1 * 1024 * 1024

class Meshes(Linter):

    def __init__(self, file: Path, settings: dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Finds issues in model files, such as incorrect file format or too large size '
        super().__init__(file, settings)
        self._max_file_size = self._settings.get('diagnostic-mesh-file-size', MAX_MESH_FILE_SIZE)

    def check(self) -> Iterator[Diagnostic]:
        if False:
            for i in range(10):
                print('nop')
        if self._settings['checks'].get('diagnostic-mesh-file-extension', False):
            for check in self.checkFileFormat():
                yield check
        if self._settings['checks'].get('diagnostic-mesh-file-size', False):
            for check in self.checkFileSize():
                yield check
        yield

    def checkFileFormat(self) -> Iterator[Diagnostic]:
        if False:
            i = 10
            return i + 15
        ' Check if mesh is in supported format '
        if self._file.suffix.lower() not in ('.3mf', '.obj', '.stl'):
            yield Diagnostic(file=self._file, diagnostic_name='diagnostic-mesh-file-extension', message=f'Extension {self._file.suffix} not supported, use 3mf, obj or stl', level='Error', offset=1)
        yield

    def checkFileSize(self) -> Iterator[Diagnostic]:
        if False:
            i = 10
            return i + 15
        ' Check if file is within size limits for Cura '
        if self._file.stat().st_size > self._max_file_size:
            yield Diagnostic(file=self._file, diagnostic_name='diagnostic-mesh-file-size', message=f'Mesh file with a size {self._file.stat().st_size} is bigger then allowed maximum of {self._max_file_size}', level='Error', offset=1)
        yield