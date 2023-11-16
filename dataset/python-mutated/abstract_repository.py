from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from poetry.core.constraints.version import Version
    from poetry.core.packages.dependency import Dependency
    from poetry.core.packages.package import Package

class AbstractRepository(ABC):

    def __init__(self, name: str) -> None:
        if False:
            print('Hello World!')
        self._name = name

    @property
    def name(self) -> str:
        if False:
            return 10
        return self._name

    @abstractmethod
    def find_packages(self, dependency: Dependency) -> list[Package]:
        if False:
            return 10
        ...

    @abstractmethod
    def search(self, query: str) -> list[Package]:
        if False:
            print('Hello World!')
        ...

    @abstractmethod
    def package(self, name: str, version: Version, extras: list[str] | None=None) -> Package:
        if False:
            for i in range(10):
                print('nop')
        ...