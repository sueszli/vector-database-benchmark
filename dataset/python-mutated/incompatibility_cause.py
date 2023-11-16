from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from poetry.mixology.incompatibility import Incompatibility

class IncompatibilityCause(Exception):
    """
    The reason and Incompatibility's terms are incompatible.
    """

class RootCause(IncompatibilityCause):
    pass

class NoVersionsCause(IncompatibilityCause):
    pass

class DependencyCause(IncompatibilityCause):
    pass

class ConflictCause(IncompatibilityCause):
    """
    The incompatibility was derived from two existing incompatibilities
    during conflict resolution.
    """

    def __init__(self, conflict: Incompatibility, other: Incompatibility) -> None:
        if False:
            return 10
        self._conflict = conflict
        self._other = other

    @property
    def conflict(self) -> Incompatibility:
        if False:
            for i in range(10):
                print('nop')
        return self._conflict

    @property
    def other(self) -> Incompatibility:
        if False:
            while True:
                i = 10
        return self._other

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return str(self._conflict)

class PythonCause(IncompatibilityCause):
    """
    The incompatibility represents a package's python constraint
    (Python versions) being incompatible
    with the current python version.
    """

    def __init__(self, python_version: str, root_python_version: str) -> None:
        if False:
            return 10
        self._python_version = python_version
        self._root_python_version = root_python_version

    @property
    def python_version(self) -> str:
        if False:
            while True:
                i = 10
        return self._python_version

    @property
    def root_python_version(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._root_python_version

class PlatformCause(IncompatibilityCause):
    """
    The incompatibility represents a package's platform constraint
    (OS most likely) being incompatible with the current platform.
    """

    def __init__(self, platform: str) -> None:
        if False:
            return 10
        self._platform = platform

    @property
    def platform(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._platform