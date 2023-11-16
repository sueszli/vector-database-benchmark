import _imp
import importlib
import sys
import zipimport
from importlib.machinery import BYTECODE_SUFFIXES, ExtensionFileLoader, FileFinder, SOURCE_SUFFIXES, SourceFileLoader, SourcelessFileLoader
from . import compile as python_compile

class PySourceFileLoader(SourceFileLoader):

    def source_to_code(self, data, path, *, _optimize=-1):
        if False:
            i = 10
            return i + 15
        'Similar to SourceFileLoader.source_to_code\n        but use the python based bytecode generator from\n        Lib/compiler/pycodegen.py\n        '
        return importlib._bootstrap._call_with_frames_removed(python_compile, data, path, 'exec', optimize=_optimize)

def _install_source_loader_helper(source_loader_type):
    if False:
        while True:
            i = 10
    extensions = (ExtensionFileLoader, _imp.extension_suffixes())
    source = (source_loader_type, SOURCE_SUFFIXES)
    bytecode = (SourcelessFileLoader, BYTECODE_SUFFIXES)
    supported_loaders = [extensions, source, bytecode]
    sys.path_hooks[:] = [zipimport.zipimporter, FileFinder.path_hook(*supported_loaders)]
    sys.path_importer_cache.clear()

def _install_py_loader():
    if False:
        return 10
    _install_source_loader_helper(PySourceFileLoader)

def _install_strict_loader():
    if False:
        i = 10
        return i + 15
    from .strict.loader import install
    install()