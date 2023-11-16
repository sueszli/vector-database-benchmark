from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from poetry.core.packages.dependency import Dependency
    from poetry.core.packages.package import Package

class DependencyPackage:

    def __init__(self, dependency: Dependency, package: Package) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._dependency = dependency
        self._package = package

    @property
    def dependency(self) -> Dependency:
        if False:
            while True:
                i = 10
        return self._dependency

    @property
    def package(self) -> Package:
        if False:
            for i in range(10):
                print('nop')
        return self._package

    def clone(self) -> DependencyPackage:
        if False:
            print('Hello World!')
        return self.__class__(self._dependency, self._package.clone())

    def with_features(self, features: list[str]) -> DependencyPackage:
        if False:
            return 10
        return self.__class__(self._dependency, self._package.with_features(features))

    def without_features(self) -> DependencyPackage:
        if False:
            for i in range(10):
                print('nop')
        return self.with_features([])

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return str(self._package)

    def __repr__(self) -> str:
        if False:
            return 10
        return repr(self._package)

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash(self._package)

    def __eq__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        if isinstance(other, DependencyPackage):
            other = other.package
        equal: bool = self._package == other
        return equal