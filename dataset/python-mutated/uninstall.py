from __future__ import annotations
from typing import TYPE_CHECKING
from poetry.installation.operations.operation import Operation
if TYPE_CHECKING:
    from poetry.core.packages.package import Package

class Uninstall(Operation):

    def __init__(self, package: Package, reason: str | None=None, priority: float | int=float('inf')) -> None:
        if False:
            print('Hello World!')
        super().__init__(reason, priority=priority)
        self._package = package

    @property
    def package(self) -> Package:
        if False:
            while True:
                i = 10
        return self._package

    @property
    def job_type(self) -> str:
        if False:
            return 10
        return 'uninstall'

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'Uninstalling {self.package.pretty_name} ({self.format_version(self._package)})'

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<Uninstall {self.package.pretty_name} ({self.format_version(self.package)})>'