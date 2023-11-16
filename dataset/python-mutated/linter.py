from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator
from ..diagnostic import Diagnostic

class Linter(ABC):

    def __init__(self, file: Path, settings: dict) -> None:
        if False:
            while True:
                i = 10
        ' Yields Diagnostics for file, these are issues with the file such as bad text format or too large file size.\n\n        @param file: A file to generate diagnostics for\n        @param settings: A list of settings containing rules for creating diagnostics\n        '
        self._settings = settings
        self._file = file

    @abstractmethod
    def check(self) -> Iterator[Diagnostic]:
        if False:
            for i in range(10):
                print('nop')
        pass