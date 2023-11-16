"""
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import ClassVar
from ...util.dataclasses import dataclass
__all__ = ()

@dataclass(frozen=True)
class Issue:
    code: int
    name: str
    description: str

@dataclass(frozen=True)
class Warning(Issue):
    _code_map: ClassVar[dict[int, Warning]] = {}
    _name_map: ClassVar[dict[str, Warning]] = {}

    def __post_init__(self) -> None:
        if False:
            print('Hello World!')
        Warning._code_map[self.code] = self
        Warning._name_map[self.name] = self

    @classmethod
    def get_by_code(cls, code: int) -> Warning:
        if False:
            for i in range(10):
                print('nop')
        return cls._code_map[code]

    @classmethod
    def get_by_name(cls, name: str) -> Warning:
        if False:
            while True:
                i = 10
        return cls._name_map[name]

    @classmethod
    def all(cls) -> list[Warning]:
        if False:
            i = 10
            return i + 15
        return list(cls._code_map.values())

@dataclass(frozen=True)
class Error(Issue):
    _code_map: ClassVar[dict[int, Error]] = {}
    _name_map: ClassVar[dict[str, Error]] = {}

    def __post_init__(self) -> None:
        if False:
            print('Hello World!')
        Error._code_map[self.code] = self
        Error._name_map[self.name] = self

    @classmethod
    def get_by_code(cls, code: int) -> Error:
        if False:
            i = 10
            return i + 15
        return cls._code_map[code]

    @classmethod
    def get_by_name(cls, name: str) -> Error:
        if False:
            while True:
                i = 10
        return cls._name_map[name]

    @classmethod
    def all(cls) -> list[Error]:
        if False:
            i = 10
            return i + 15
        return list(cls._code_map.values())