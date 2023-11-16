""" This module abstracts what site.py is normally doing in .pth files.

This tries to extract "namespaces" packages that were manually created and
point to package directories, which need no "__init__.py" to count as a
package. Nuitka will pretend for those that there be one, but without content.
"""
import os
import sys
from nuitka.Tracing import recursion_logger
from nuitka.utils.FileOperations import getFileContentByLine, listDir
from nuitka.utils.ModuleNames import ModuleName

def getLoadedPackages():
    if False:
        print('Hello World!')
    'Extract packages with no __file__, i.e. they got added manually.\n\n    They are frequently created with "*.pth" files that then check for the\n    "__init__.py" to exist, and when it doesn\'t, then they create during the\n    loading of "site.py" an package with "__path__" set.\n    '
    for (module_name, module) in sys.modules.items():
        if not getattr(module, '__path__', None):
            continue
        if hasattr(module, '__file__'):
            continue
        yield (module_name, module)

def detectPreLoadedPackagePaths():
    if False:
        i = 10
        return i + 15
    result = {}
    for (package_name, module) in getLoadedPackages():
        result[package_name] = list(module.__path__)
    return result
preloaded_packages = None

def getPreloadedPackagePaths():
    if False:
        i = 10
        return i + 15
    'Return dictionary with preloaded package paths from .pth files'
    global preloaded_packages
    if preloaded_packages is None:
        preloaded_packages = detectPreLoadedPackagePaths()
    return preloaded_packages

def setPreloadedPackagePaths(value):
    if False:
        while True:
            i = 10
    global preloaded_packages
    preloaded_packages = value

def getPreloadedPackagePath(package_name):
    if False:
        for i in range(10):
            print('nop')
    return getPreloadedPackagePaths().get(package_name)

def isPreloadedPackagePath(path):
    if False:
        i = 10
        return i + 15
    path = os.path.normcase(path)
    for paths in getPreloadedPackagePaths().values():
        for element in paths:
            if os.path.normcase(element) == path:
                return True
    return False

def _considerPthImportedPackage(module_name):
    if False:
        print('Hello World!')
    if module_name in ('os', 'sys'):
        return None
    if module_name.startswith('__editable__'):
        finder_module = __import__(module_name)
        paths = set()
        mapping = getattr(finder_module, 'MAPPING', {})
        for (package_name, path) in mapping.items():
            if os.path.basename(path) != package_name:
                continue
            paths.add(os.path.dirname(path))
        sys.path.extend(sorted(paths))
        return None
    return module_name

def detectPthImportedPackages():
    if False:
        for i in range(10):
            print('nop')
    if not hasattr(sys.modules['site'], 'getsitepackages'):
        return ()
    pth_imports = set()
    for prefix in sys.modules['site'].getsitepackages():
        if not os.path.isdir(prefix):
            continue
        for (path, filename) in listDir(prefix):
            if filename.endswith('.pth'):
                try:
                    for line in getFileContentByLine(path, 'rU'):
                        if line.startswith('import '):
                            if ';' in line:
                                line = line[:line.find(';')]
                            for part in line[7:].split(','):
                                pth_import = _considerPthImportedPackage(part.strip())
                                if pth_import is not None:
                                    pth_imports.add(pth_import)
                except OSError:
                    recursion_logger.warning("Python installation problem, cannot read file '%s'.")
    return tuple(sorted(pth_imports))
pth_imported_packages = ()

def setPthImportedPackages(value):
    if False:
        i = 10
        return i + 15
    global pth_imported_packages
    pth_imported_packages = tuple((ModuleName(module_name) for module_name in value))

def getPthImportedPackages():
    if False:
        print('Hello World!')
    return pth_imported_packages