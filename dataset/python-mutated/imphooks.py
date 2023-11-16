"""Import hooks for importing xonsh source files.

This module registers the hooks it defines when it is imported.
"""
import contextlib
import importlib
import os
import re
import sys
import types
from importlib.abc import Loader, MetaPathFinder, SourceLoader
from importlib.machinery import ModuleSpec
from xonsh.built_ins import XSH
from xonsh.events import events
from xonsh.execer import Execer
from xonsh.lazyasd import lazyobject
from xonsh.platform import ON_WINDOWS
from xonsh.tools import print_warning

@lazyobject
def ENCODING_LINE():
    if False:
        while True:
            i = 10
    return re.compile(b'^[ \t\x0c]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)')

def find_source_encoding(src):
    if False:
        i = 10
        return i + 15
    'Finds the source encoding given bytes representing a file by checking\n    a special comment at either the first or second line of the source file.\n    https://docs.python.org/3/howto/unicode.html#unicode-literals-in-python-source-code\n    If no encoding is found, UTF-8 codec with BOM signature will be returned\n    as it skips an optional UTF-8 encoded BOM at the start of the data\n    and is otherwise the same as UTF-8\n    https://docs.python.org/3/library/codecs.html#module-encodings.utf_8_sig\n    '
    utf8 = 'utf-8-sig'
    (first, _, rest) = src.partition(b'\n')
    m = ENCODING_LINE.match(first)
    if m is not None:
        return m.group(1).decode(utf8)
    (second, _, _) = rest.partition(b'\n')
    m = ENCODING_LINE.match(second)
    if m is not None:
        return m.group(1).decode(utf8)
    return utf8

class XonshImportHook(MetaPathFinder, SourceLoader):
    """Implements the import hook for xonsh source files."""

    def __init__(self, execer, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self._filenames = {}
        self._execer = execer

    def find_spec(self, fullname, path, target=None):
        if False:
            for i in range(10):
                print('nop')
        'Finds the spec for a xonsh module if it exists.'
        dot = '.'
        spec = None
        path = sys.path if path is None else path
        if dot not in fullname and dot not in path:
            path = [dot] + path
        name = fullname.rsplit(dot, 1)[-1]
        fname = name + '.xsh'
        for p in path:
            if not isinstance(p, str):
                continue
            if not os.path.isdir(p) or not os.access(p, os.R_OK):
                continue
            if fname not in {x.name for x in os.scandir(p)}:
                continue
            spec = ModuleSpec(fullname, self)
            self._filenames[fullname] = os.path.abspath(os.path.join(p, fname))
            break
        return spec

    def create_module(self, spec):
        if False:
            while True:
                i = 10
        'Create a xonsh module with the appropriate attributes.'
        mod = types.ModuleType(spec.name)
        mod.__file__ = self.get_filename(spec.name)
        mod.__loader__ = self
        mod.__package__ = spec.parent or ''
        return mod

    def get_filename(self, fullname):
        if False:
            print('Hello World!')
        "Returns the filename for a module's fullname."
        return self._filenames[fullname]

    def get_data(self, path):
        if False:
            i = 10
            return i + 15
        'Gets the bytes for a path.'
        raise OSError

    def get_code(self, fullname):
        if False:
            for i in range(10):
                print('nop')
        'Gets the code object for a xonsh file.'
        filename = self.get_filename(fullname)
        if filename is None:
            msg = f'xonsh file {fullname!r} could not be found'
            raise ImportError(msg)
        src = self.get_source(fullname)
        execer = self._execer
        execer.filename = filename
        ctx = {}
        code = execer.compile(src, glbs=ctx, locs=ctx)
        return code

    def get_source(self, fullname):
        if False:
            i = 10
            return i + 15
        if fullname is None:
            raise ImportError('could not find fullname to module')
        filename = self.get_filename(fullname)
        with open(filename, 'rb') as f:
            src = f.read()
        if ON_WINDOWS:
            src = src.replace(b'\r\n', b'\n')
        enc = find_source_encoding(src)
        src = src.decode(encoding=enc)
        src = src if src.endswith('\n') else src + '\n'
        return src
events.doc('on_import_pre_find_spec', '\non_import_pre_find_spec(fullname: str, path: str, target: module or None) -> None\n\nFires before any import find_spec() calls have been executed. The parameters\nhere are the same as importlib.abc.MetaPathFinder.find_spec(). Namely,\n\n:``fullname``: The full name of the module to import.\n:``path``: None if a top-level import, otherwise the ``__path__`` of the parent\n          package.\n:``target``: Target module used to make a better guess about the package spec.\n')
events.doc('on_import_post_find_spec', '\non_import_post_find_spec(spec, fullname, path, target) -> None\n\nFires after all import find_spec() calls have been executed. The parameters\nhere the spec and the arguments importlib.abc.MetaPathFinder.find_spec(). Namely,\n\n:``spec``: A ModuleSpec object if the spec was found, or None if it was not.\n:``fullname``: The full name of the module to import.\n:``path``: None if a top-level import, otherwise the ``__path__`` of the parent\n          package.\n:``target``: Target module used to make a better guess about the package spec.\n')
events.doc('on_import_pre_create_module', '\non_import_pre_create_module(spec: ModuleSpec) -> None\n\nFires right before a module is created by its loader. The only parameter\nis the spec object. See importlib for more details.\n')
events.doc('on_import_post_create_module', '\non_import_post_create_module(module: Module, spec: ModuleSpec) -> None\n\nFires after a module is created by its loader but before the loader returns it.\nThe parameters here are the module object itself and the spec object.\nSee importlib for more details.\n')
events.doc('on_import_pre_exec_module', '\non_import_pre_exec_module(module: Module) -> None\n\nFires right before a module is executed by its loader. The only parameter\nis the module itself. See importlib for more details.\n')
events.doc('on_import_post_exec_module', '\non_import_post_create_module(module: Module) -> None\n\nFires after a module is executed by its loader but before the loader returns it.\nThe only parameter is the module itself. See importlib for more details.\n')

def _should_dispatch_xonsh_import_event_loader():
    if False:
        for i in range(10):
            print('nop')
    'Figures out if we should dispatch to a load event'
    return len(events.on_import_pre_create_module) > 0 or len(events.on_import_post_create_module) > 0 or len(events.on_import_pre_exec_module) > 0 or (len(events.on_import_post_exec_module) > 0)

class XonshImportEventHook(MetaPathFinder):
    """Implements the import hook for firing xonsh events on import."""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self._fullname_stack = []

    @contextlib.contextmanager
    def append_stack(self, fullname):
        if False:
            print('Hello World!')
        'A context manager for appending and then removing a name from the\n        fullname stack.\n        '
        self._fullname_stack.append(fullname)
        yield
        del self._fullname_stack[-1]

    def find_spec(self, fullname, path, target=None):
        if False:
            while True:
                i = 10
        'Finds the spec for a xonsh module if it exists.'
        if fullname in reversed(self._fullname_stack):
            return None
        npre = len(events.on_import_pre_find_spec)
        npost = len(events.on_import_post_find_spec)
        dispatch_load = _should_dispatch_xonsh_import_event_loader()
        if npre > 0:
            events.on_import_pre_find_spec.fire(fullname=fullname, path=path, target=target)
        elif npost == 0 and (not dispatch_load):
            return None
        with self.append_stack(fullname):
            spec = importlib.util.find_spec(fullname)
        if npost > 0:
            events.on_import_post_find_spec.fire(spec=spec, fullname=fullname, path=path, target=target)
        if dispatch_load and spec is not None and hasattr(spec.loader, 'create_module'):
            spec.loader = XonshImportEventLoader(spec.loader)
        return spec
_XIEVL_WRAPPED_ATTRIBUTES = frozenset(['load_module', 'module_repr', 'get_data', 'get_resource_filename', 'get_resource_stream', 'get_resource_string', 'has_resource', 'has_metadata', 'get_metadata', 'get_metadata_lines', 'resource_isdir', 'metadata_isdir', 'resource_listdir', 'metadata_listdir'])

class XonshImportEventLoader(Loader):
    """A class that dispatches loader calls to another loader and fires relevant
    xonsh events.
    """

    def __init__(self, loader):
        if False:
            while True:
                i = 10
        self.loader = loader

    def create_module(self, spec):
        if False:
            for i in range(10):
                print('nop')
        'Creates and returns the module object.'
        events.on_import_pre_create_module.fire(spec=spec)
        mod = self.loader.create_module(spec)
        events.on_import_post_create_module.fire(module=mod, spec=spec)
        return mod

    def exec_module(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Executes the module in its own namespace.'
        events.on_import_pre_exec_module.fire(module=module)
        rtn = self.loader.exec_module(module)
        events.on_import_post_exec_module.fire(module=module)
        return rtn

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name in _XIEVL_WRAPPED_ATTRIBUTES:
            return getattr(self.loader, name)
        return object.__getattribute__(self, name)
ARG_NOT_PRESENT = object()

def install_import_hooks(execer=ARG_NOT_PRESENT):
    if False:
        for i in range(10):
            print('nop')
    '\n    Install Xonsh import hooks in ``sys.meta_path`` in order for ``.xsh`` files\n    to be importable and import events to be fired.\n\n    Can safely be called many times, will be no-op if xonsh import hooks are\n    already present.\n    '
    if execer is ARG_NOT_PRESENT:
        print_warning('No execer was passed to install_import_hooks. This will become an error in future.')
        execer = XSH.execer
        if execer is None:
            execer = Execer()
            XSH.load(execer=execer)
    found_imp = found_event = False
    for hook in sys.meta_path:
        if isinstance(hook, XonshImportHook):
            found_imp = True
        elif isinstance(hook, XonshImportEventHook):
            found_event = True
    if not found_imp:
        sys.meta_path.append(XonshImportHook(execer))
    if not found_event:
        sys.meta_path.insert(0, XonshImportEventHook())
install_hook = install_import_hooks