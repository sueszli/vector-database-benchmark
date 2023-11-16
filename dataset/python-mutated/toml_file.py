from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import tomlkit
from tomlkit.toml_document import TOMLDocument
from tomlkit.toml_file import TOMLFile
from pdm import termui

class TOMLBase(TOMLFile):

    def __init__(self, path: str | Path, *, ui: termui.UI) -> None:
        if False:
            while True:
                i = 10
        super().__init__(path)
        self._path = Path(path)
        self.ui = ui
        self._data = self.read()

    def read(self) -> TOMLDocument:
        if False:
            print('Hello World!')
        if not self._path.exists():
            return tomlkit.document()
        return super().read()

    def set_data(self, data: Mapping[str, Any]) -> None:
        if False:
            print('Hello World!')
        'Set the data of the TOML file.'
        self._data = tomlkit.document()
        self._data.update(data)

    def reload(self) -> None:
        if False:
            return 10
        self._data = self.read()

    def write(self) -> None:
        if False:
            print('Hello World!')
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return super().write(self._data)

    def exists(self) -> bool:
        if False:
            return 10
        return self._path.exists()

    def empty(self) -> bool:
        if False:
            return 10
        return not self._data