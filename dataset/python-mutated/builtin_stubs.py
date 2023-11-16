"""Utilities for parsing pytd files for builtins."""
from pytype import pytype_source_utils
from pytype.imports import base
from pytype.platform_utils import path_utils
from pytype.pyi import parser
from pytype.pytd import visitors
_cached_builtins_pytd = []

def InvalidateCache():
    if False:
        return 10
    if _cached_builtins_pytd:
        del _cached_builtins_pytd[0]

def GetBuiltinsAndTyping(options):
    if False:
        i = 10
        return i + 15
    if not _cached_builtins_pytd:
        _cached_builtins_pytd.append(BuiltinsAndTyping().load(options))
    return _cached_builtins_pytd[0]
DEFAULT_SRC = '\nfrom typing import Any\ndef __getattr__(name: Any) -> Any: ...\n'

def GetDefaultAst(options):
    if False:
        for i in range(10):
            print('nop')
    return parser.parse_string(src=DEFAULT_SRC, options=options)

class BuiltinsAndTyping:
    """The builtins and typing modules, which need to be treated specially."""

    def _parse_predefined(self, name, options):
        if False:
            print('Hello World!')
        (_, src) = GetPredefinedFile('builtins', name, '.pytd')
        mod = parser.parse_string(src, name=name, options=options)
        return mod

    def load(self, options):
        if False:
            for i in range(10):
                print('nop')
        'Read builtins.pytd and typing.pytd, and return the parsed modules.'
        t = self._parse_predefined('typing', options)
        b = self._parse_predefined('builtins', options)
        b = b.Visit(visitors.LookupExternalTypes({'typing': t}, self_name='builtins'))
        t = t.Visit(visitors.LookupBuiltins(b))
        b = b.Visit(visitors.NamedTypeToClassType())
        t = t.Visit(visitors.NamedTypeToClassType())
        b = b.Visit(visitors.AdjustTypeParameters())
        t = t.Visit(visitors.AdjustTypeParameters())
        b = b.Visit(visitors.CanonicalOrderingVisitor())
        t = t.Visit(visitors.CanonicalOrderingVisitor())
        b.Visit(visitors.FillInLocalPointers({'': b, 'typing': t, 'builtins': b}))
        t.Visit(visitors.FillInLocalPointers({'': t, 'typing': t, 'builtins': b}))
        b.Visit(visitors.VerifyLookup())
        t.Visit(visitors.VerifyLookup())
        b.Visit(visitors.VerifyContainers())
        t.Visit(visitors.VerifyContainers())
        return (b, t)

def GetPredefinedFile(stubs_subdir, module, extension='.pytd', as_package=False):
    if False:
        while True:
            i = 10
    'Get the contents of a predefined PyTD, typically with a file name *.pytd.\n\n  Arguments:\n    stubs_subdir: the directory, typically "builtins" or "stdlib"\n    module: module name (e.g., "sys" or "__builtins__")\n    extension: either ".pytd" or ".py"\n    as_package: try the module as a directory with an __init__ file\n  Returns:\n    The contents of the file\n  Raises:\n    IOError: if file not found\n  '
    parts = module.split('.')
    if as_package:
        parts.append('__init__')
    mod_path = path_utils.join(*parts) + extension
    path = path_utils.join('stubs', stubs_subdir, mod_path)
    return (path, pytype_source_utils.load_text_file(path))

class BuiltinLoader(base.BuiltinLoader):
    """Load builtins from the pytype source tree."""

    def __init__(self, options):
        if False:
            while True:
                i = 10
        self.options = options

    def _parse_predefined(self, pytd_subdir, module, as_package=False):
        if False:
            for i in range(10):
                print('nop')
        'Parse a pyi/pytd file in the pytype source tree.'
        try:
            (filename, src) = GetPredefinedFile(pytd_subdir, module, as_package=as_package)
        except OSError:
            return None
        ast = parser.parse_string(src, filename=filename, name=module, options=self.options)
        assert ast.name == module
        return ast

    def load_module(self, namespace, module_name):
        if False:
            print('Hello World!')
        'Load a stub that ships with pytype.'
        mod = self._parse_predefined(namespace, module_name)
        if mod:
            filename = module_name
        else:
            mod = self._parse_predefined(namespace, module_name, as_package=True)
            filename = path_utils.join(module_name, '__init__.pyi')
        return (filename, mod)