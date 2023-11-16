"""
Facilities for helping test code which interacts with Python's module system
to load code.
"""
import sys
from types import ModuleType
from typing import Iterable, List, Tuple
from twisted.python.filepath import FilePath

class TwistedModulesMixin:
    """
    A mixin for C{twisted.trial.unittest.SynchronousTestCase} providing useful
    methods for manipulating Python's module system.
    """

    def replaceSysPath(self, sysPath: List[str]) -> None:
        if False:
            return 10
        '\n        Replace sys.path, for the duration of the test, with the given value.\n        '
        originalSysPath = sys.path[:]

        def cleanUpSysPath() -> None:
            if False:
                for i in range(10):
                    print('nop')
            sys.path[:] = originalSysPath
        self.addCleanup(cleanUpSysPath)
        sys.path[:] = sysPath

    def replaceSysModules(self, sysModules: Iterable[Tuple[str, ModuleType]]) -> None:
        if False:
            while True:
                i = 10
        '\n        Replace sys.modules, for the duration of the test, with the given value.\n        '
        originalSysModules = sys.modules.copy()

        def cleanUpSysModules() -> None:
            if False:
                i = 10
                return i + 15
            sys.modules.clear()
            sys.modules.update(originalSysModules)
        self.addCleanup(cleanUpSysModules)
        sys.modules.clear()
        sys.modules.update(sysModules)

    def pathEntryWithOnePackage(self, pkgname: str='test_package') -> FilePath[str]:
        if False:
            return 10
        '\n        Generate a L{FilePath} with one package, named C{pkgname}, on it, and\n        return the L{FilePath} of the path entry.\n        '
        entry = FilePath(self.mktemp())
        pkg = entry.child('test_package')
        pkg.makedirs()
        pkg.child('__init__.py').setContent(b'')
        return entry