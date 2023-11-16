from abc import ABC, abstractmethod
from pathlib import Path

class FileFormatter(ABC):

    def __init__(self, settings: dict) -> None:
        if False:
            return 10
        ' Yields Diagnostics for file, these are issues with the file such as bad text format or too large file size.\n\n        @param file: A file to generate diagnostics for\n        @param settings: A list of settings containing rules for creating diagnostics\n        '
        self._settings = settings

    @abstractmethod
    def formatFile(self, file: Path) -> None:
        if False:
            return 10
        pass