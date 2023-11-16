from __future__ import annotations
from typing import TYPE_CHECKING
from poetry.mixology.term import Term
if TYPE_CHECKING:
    from poetry.core.packages.dependency import Dependency
    from poetry.core.packages.package import Package
    from poetry.mixology.incompatibility import Incompatibility

class Assignment(Term):
    """
    A term in a PartialSolution that tracks some additional metadata.
    """

    def __init__(self, dependency: Dependency, is_positive: bool, decision_level: int, index: int, cause: Incompatibility | None=None) -> None:
        if False:
            return 10
        super().__init__(dependency, is_positive)
        self._decision_level = decision_level
        self._index = index
        self._cause = cause

    @property
    def decision_level(self) -> int:
        if False:
            print('Hello World!')
        return self._decision_level

    @property
    def index(self) -> int:
        if False:
            while True:
                i = 10
        return self._index

    @property
    def cause(self) -> Incompatibility | None:
        if False:
            print('Hello World!')
        return self._cause

    @classmethod
    def decision(cls, package: Package, decision_level: int, index: int) -> Assignment:
        if False:
            return 10
        return cls(package.to_dependency(), True, decision_level, index)

    @classmethod
    def derivation(cls, dependency: Dependency, is_positive: bool, cause: Incompatibility, decision_level: int, index: int) -> Assignment:
        if False:
            while True:
                i = 10
        return cls(dependency, is_positive, decision_level, index, cause)

    def is_decision(self) -> bool:
        if False:
            print('Hello World!')
        return self._cause is None