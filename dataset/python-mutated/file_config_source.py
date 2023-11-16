from __future__ import annotations
from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import Any
from tomlkit import document
from tomlkit import table
from poetry.config.config_source import ConfigSource
if TYPE_CHECKING:
    from collections.abc import Iterator
    from tomlkit.toml_document import TOMLDocument
    from poetry.toml.file import TOMLFile

class FileConfigSource(ConfigSource):

    def __init__(self, file: TOMLFile, auth_config: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        self._file = file
        self._auth_config = auth_config

    @property
    def name(self) -> str:
        if False:
            return 10
        return str(self._file.path)

    @property
    def file(self) -> TOMLFile:
        if False:
            print('Hello World!')
        return self._file

    def add_property(self, key: str, value: Any) -> None:
        if False:
            while True:
                i = 10
        with self.secure() as toml:
            config: dict[str, Any] = toml
            keys = key.split('.')
            for (i, key) in enumerate(keys):
                if key not in config and i < len(keys) - 1:
                    config[key] = table()
                if i == len(keys) - 1:
                    config[key] = value
                    break
                config = config[key]

    def remove_property(self, key: str) -> None:
        if False:
            while True:
                i = 10
        with self.secure() as toml:
            config: dict[str, Any] = toml
            keys = key.split('.')
            current_config = config
            for (i, key) in enumerate(keys):
                if key not in current_config:
                    return
                if i == len(keys) - 1:
                    del current_config[key]
                    break
                current_config = current_config[key]

    @contextmanager
    def secure(self) -> Iterator[TOMLDocument]:
        if False:
            return 10
        if self.file.exists():
            initial_config = self.file.read()
            config = self.file.read()
        else:
            initial_config = document()
            config = document()
        new_file = not self.file.exists()
        yield config
        try:
            mode = 384
            if new_file:
                self.file.path.touch(mode=mode)
            self.file.write(config)
        except Exception:
            self.file.write(initial_config)
            raise