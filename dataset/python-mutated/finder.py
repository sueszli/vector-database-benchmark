from __future__ import annotations
from typing import Any
import unearth
from packaging.version import Version
from unearth.evaluator import Package
from unearth.session import PyPISession

class ReverseVersion(Version):
    """A subclass of version that reverse the order of comparison."""

    def __lt__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        return super().__gt__(other)

    def __le__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return super().__ge__(other)

    def __gt__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        return super().__lt__(other)

    def __ge__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return super().__le__(other)

class PDMPackageFinder(unearth.PackageFinder):

    def __init__(self, session: PyPISession | None=None, *, minimal_version: bool=False, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self.minimal_version = minimal_version
        super().__init__(session, **kwargs)

    def _sort_key(self, package: Package) -> tuple:
        if False:
            i = 10
            return i + 15
        key = super()._sort_key(package)
        if self.minimal_version:
            key_list = list(key)
            key_list[2] = ReverseVersion(package.version) if package.version else ReverseVersion('0')
            key = tuple(key_list)
        return key