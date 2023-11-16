from __future__ import annotations
from typing import TYPE_CHECKING
from typing import TypeVar
if TYPE_CHECKING:
    from poetry.core.packages.package import Package
T = TypeVar('T', bound='Operation')

class Operation:

    def __init__(self, reason: str | None=None, priority: int | float=0) -> None:
        if False:
            i = 10
            return i + 15
        self._reason = reason
        self._skipped = False
        self._skip_reason: str | None = None
        self._priority = priority

    @property
    def job_type(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @property
    def reason(self) -> str | None:
        if False:
            return 10
        return self._reason

    @property
    def skipped(self) -> bool:
        if False:
            print('Hello World!')
        return self._skipped

    @property
    def skip_reason(self) -> str | None:
        if False:
            return 10
        return self._skip_reason

    @property
    def priority(self) -> float | int:
        if False:
            print('Hello World!')
        return self._priority

    @property
    def package(self) -> Package:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def format_version(self, package: Package) -> str:
        if False:
            return 10
        version: str = package.full_pretty_version
        return version

    def skip(self: T, reason: str) -> T:
        if False:
            while True:
                i = 10
        self._skipped = True
        self._skip_reason = reason
        return self

    def unskip(self: T) -> T:
        if False:
            i = 10
            return i + 15
        self._skipped = False
        self._skip_reason = None
        return self