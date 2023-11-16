"""
Find modules used by a script, using bytecode analysis.

Based on the stdlib modulefinder by Thomas Heller and Just van Rossum,
but uses a graph data structure and 2.3 features

XXX: Verify all calls to _import_hook (and variants) to ensure that
imports are done in the right way.
"""
import ast
import codecs
import marshal
import os
import pkgutil
import sys
import re
from collections import deque, namedtuple, defaultdict
import warnings
import importlib.util
import importlib.machinery
if sys.version_info >= (3, 10):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata
from altgraph.ObjectGraph import ObjectGraph
from altgraph import GraphError
from . import util
from . import zipio
from ._compat import BytesIO, pathname2url, _READ_MODE
BOM = codecs.BOM_UTF8.decode('utf-8')

class BUILTIN_MODULE:

    def is_package(fqname):
        if False:
            print('Hello World!')
        return False

class NAMESPACE_PACKAGE:

    def __init__(self, namespace_dirs):
        if False:
            i = 10
            return i + 15
        self.namespace_dirs = namespace_dirs

    def is_package(self, fqname):
        if False:
            return 10
        return True
ABSOLUTE_OR_RELATIVE_IMPORT_LEVEL = -1
'\nConstant instructing the builtin `__import__()` function to attempt both\nabsolute and relative imports.\n'
ABSOLUTE_IMPORT_LEVEL = 0
'\nConstant instructing the builtin `__import__()` function to attempt only\nabsolute imports.\n'
DEFAULT_IMPORT_LEVEL = ABSOLUTE_OR_RELATIVE_IMPORT_LEVEL if sys.version_info[0] == 2 else ABSOLUTE_IMPORT_LEVEL
'\nConstant instructing the builtin `__import__()` function to attempt the default\nimport style specific to the active Python interpreter.\n\nSpecifically, under:\n\n* Python 2, this defaults to attempting both absolute and relative imports.\n* Python 3, this defaults to attempting only absolute imports.\n'
_IMPORTABLE_FILETYPE_EXTS = sorted(importlib.machinery.all_suffixes(), key=lambda p: len(p), reverse=True)
_packagePathMap = {}

class InvalidRelativeImportError(ImportError):
    pass
_strs = re.compile('^\\s*["\']([A-Za-z0-9_]+)["\'],?\\s*')

def _eval_str_tuple(value):
    if False:
        print('Hello World!')
    '\n    Input is the repr of a tuple of strings, output\n    is that tuple.\n\n    This only works with a tuple where the members are\n    python identifiers.\n    '
    if not (value.startswith('(') and value.endswith(')')):
        raise ValueError(value)
    orig_value = value
    value = value[1:-1]
    result = []
    while value:
        m = _strs.match(value)
        if m is None:
            raise ValueError(orig_value)
        result.append(m.group(1))
        value = value[len(m.group(0)):]
    return tuple(result)

def _path_from_importerror(exc, default):
    if False:
        print('Hello World!')
    m = re.match('^No module named (\\S+)$', str(exc))
    if m is not None:
        return m.group(1)
    return default

def os_listdir(path):
    if False:
        return 10
    '\n    Deprecated name\n    '
    warnings.warn('Use zipio.listdir instead of os_listdir', DeprecationWarning)
    return zipio.listdir(path)

def _code_to_file(co):
    if False:
        return 10
    ' Convert code object to a .pyc pseudo-file '
    if sys.version_info >= (3, 7):
        header = importlib.util.MAGIC_NUMBER + b'\x00' * 12
    elif sys.version_info >= (3, 4):
        header = importlib.util.MAGIC_NUMBER + b'\x00' * 8
    else:
        header = importlib.util.MAGIC_NUMBER + b'\x00' * 4
    return BytesIO(header + marshal.dumps(co))

def AddPackagePath(packagename, path):
    if False:
        return 10
    warnings.warn('Use addPackagePath instead of AddPackagePath', DeprecationWarning)
    addPackagePath(packagename, path)

def addPackagePath(packagename, path):
    if False:
        return 10
    paths = _packagePathMap.get(packagename, [])
    paths.append(path)
    _packagePathMap[packagename] = paths

class DependencyInfo(namedtuple('DependencyInfo', ['conditional', 'function', 'tryexcept', 'fromlist'])):
    __slots__ = ()

    def _merged(self, other):
        if False:
            return 10
        if not self.conditional and (not self.function) and (not self.tryexcept) or (not other.conditional and (not other.function) and (not other.tryexcept)):
            return DependencyInfo(conditional=False, function=False, tryexcept=False, fromlist=self.fromlist and other.fromlist)
        else:
            return DependencyInfo(conditional=self.conditional or other.conditional, function=self.function or other.function, tryexcept=self.tryexcept or other.tryexcept, fromlist=self.fromlist and other.fromlist)

class Node:
    """
    Abstract base class (ABC) of all objects added to a `ModuleGraph`.

    Attributes
    ----------
    code : codeobject
        Code object of the pure-Python module corresponding to this graph node
        if any _or_ `None` otherwise.
    graphident : str
        Synonym of `identifier` required by the `ObjectGraph` superclass of the
        `ModuleGraph` class. For readability, the `identifier` attribute should
        typically be used instead.
    filename : str
        Absolute path of this graph node's corresponding module, package, or C
        extension if any _or_ `None` otherwise.
    identifier : str
        Fully-qualified name of this graph node's corresponding module,
        package, or C extension.
    packagepath : str
        List of the absolute paths of all directories comprising this graph
        node's corresponding package. If this is a:
        * Non-namespace package, this list contains exactly one path.
        * Namespace package, this list contains one or more paths.
    _deferred_imports : list
        List of all target modules imported by the source module corresponding
        to this graph node whole importations have been deferred for subsequent
        processing in between calls to the `_ModuleGraph._scan_code()` and
        `_ModuleGraph._process_imports()` methods for this source module _or_
        `None` otherwise. Each element of this list is a 3-tuple
        `(have_star, _safe_import_hook_args, _safe_import_hook_kwargs)`
        collecting the importation of a target module from this source module
        for subsequent processing, where:
        * `have_star` is a boolean `True` only if this is a `from`-style star
          import (e.g., resembling `from {target_module_name} import *`).
        * `_safe_import_hook_args` is a (typically non-empty) sequence of all
          positional arguments to be passed to the `_safe_import_hook()` method
          to add this importation to the graph.
        * `_safe_import_hook_kwargs` is a (typically empty) dictionary of all
          keyword arguments to be passed to the `_safe_import_hook()` method
          to add this importation to the graph.
        Unlike functional languages, Python imposes a maximum depth on the
        interpreter stack (and hence recursion). On breaching this depth,
        Python raises a fatal `RuntimeError` exception. Since `ModuleGraph`
        parses imports recursively rather than iteratively, this depth _was_
        commonly breached before the introduction of this list. Python
        environments installing a large number of modules (e.g., Anaconda) were
        particularly susceptible. Why? Because `ModuleGraph` concurrently
        descended through both the abstract syntax trees (ASTs) of all source
        modules being parsed _and_ the graph of all target modules imported by
        these source modules being built. The stack thus consisted of
        alternating layers of AST and graph traversal. To unwind such
        alternation and effectively halve the stack depth, `ModuleGraph` now
        descends through the abstract syntax tree (AST) of each source module
        being parsed and adds all importations originating within this module
        to this list _before_ descending into the graph of these importations.
        See pyinstaller/pyinstaller/#1289 for further details.
    _global_attr_names : set
        Set of the unqualified names of all global attributes (e.g., classes,
        variables) defined in the pure-Python module corresponding to this
        graph node if any _or_ the empty set otherwise. This includes the names
        of all attributes imported via `from`-style star imports from other
        existing modules (e.g., `from {target_module_name} import *`). This
        set is principally used to differentiate the non-ignorable importation
        of non-existent submodules in a package from the ignorable importation
        of existing global attributes defined in that package's pure-Python
        `__init__` submodule in `from`-style imports (e.g., `bar` in
        `from foo import bar`, which may be either a submodule or attribute of
        `foo`), as such imports ambiguously allow both. This set is _not_ used
        to differentiate submodules from attributes in `import`-style imports
        (e.g., `bar` in `import foo.bar`, which _must_ be a submodule of
        `foo`), as such imports unambiguously allow only submodules.
    _starimported_ignored_module_names : set
        Set of the fully-qualified names of all existing unparsable modules
        that the existing parsable module corresponding to this graph node
        attempted to perform one or more "star imports" from. If this module
        either does _not_ exist or does but is unparsable, this is the empty
        set. Equivalently, this set contains each fully-qualified name
        `{trg_module_name}` for which:
        * This module contains an import statement of the form
          `from {trg_module_name} import *`.
        * The module whose name is `{trg_module_name}` exists but is _not_
          parsable by `ModuleGraph` (e.g., due to _not_ being pure-Python).
        **This set is currently defined but otherwise ignored.**
    _submodule_basename_to_node : dict
        Dictionary mapping from the unqualified name of each submodule
        contained by the parent module corresponding to this graph node to that
        submodule's graph node. If this dictionary is non-empty, this parent
        module is typically but _not_ always a package (e.g., the non-package
        `os` module containing the `os.path` submodule).
    """
    __slots__ = ['code', 'filename', 'graphident', 'identifier', 'packagepath', '_deferred_imports', '_global_attr_names', '_starimported_ignored_module_names', '_submodule_basename_to_node']

    def __init__(self, identifier):
        if False:
            print('Hello World!')
        "\n        Initialize this graph node.\n\n        Parameters\n        ----------\n        identifier : str\n            Fully-qualified name of this graph node's corresponding module,\n            package, or C extension.\n        "
        self.code = None
        self.filename = None
        self.graphident = identifier
        self.identifier = identifier
        self.packagepath = None
        self._deferred_imports = None
        self._global_attr_names = set()
        self._starimported_ignored_module_names = set()
        self._submodule_basename_to_node = dict()

    def is_global_attr(self, attr_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        `True` only if the pure-Python module corresponding to this graph node\n        defines a global attribute (e.g., class, variable) with the passed\n        name.\n\n        If this module is actually a package, this method instead returns\n        `True` only if this package\'s pure-Python `__init__` submodule defines\n        such a global attribute. In this case, note that this package may still\n        contain an importable submodule of the same name. Callers should\n        attempt to import this attribute as a submodule of this package\n        _before_ assuming this attribute to be an ignorable global. See\n        "Examples" below for further details.\n\n        Parameters\n        ----------\n        attr_name : str\n            Unqualified name of the attribute to be tested.\n\n        Returns\n        ----------\n        bool\n            `True` only if this module defines this global attribute.\n\n        Examples\n        ----------\n        Consider a hypothetical module `foo` containing submodules `bar` and\n        `__init__` where the latter assigns `bar` to be a global variable\n        (possibly star-exported via the special `__all__` global variable):\n\n        >>> # In "foo.__init__":\n        >>> bar = 3.1415\n\n        Python 2 and 3 both permissively permit this. This method returns\n        `True` in this case (i.e., when called on the `foo` package\'s graph\n        node, passed the attribute name `bar`) despite the importability of the\n        `foo.bar` submodule.\n        '
        return attr_name in self._global_attr_names

    def is_submodule(self, submodule_basename):
        if False:
            return 10
        '\n        `True` only if the parent module corresponding to this graph node\n        contains the submodule with the passed name.\n\n        If `True`, this parent module is typically but _not_ always a package\n        (e.g., the non-package `os` module containing the `os.path` submodule).\n\n        Parameters\n        ----------\n        submodule_basename : str\n            Unqualified name of the submodule to be tested.\n\n        Returns\n        ----------\n        bool\n            `True` only if this parent module contains this submodule.\n        '
        return submodule_basename in self._submodule_basename_to_node

    def add_global_attr(self, attr_name):
        if False:
            print('Hello World!')
        "\n        Record the global attribute (e.g., class, variable) with the passed\n        name to be defined by the pure-Python module corresponding to this\n        graph node.\n\n        If this module is actually a package, this method instead records this\n        attribute to be defined by this package's pure-Python `__init__`\n        submodule.\n\n        Parameters\n        ----------\n        attr_name : str\n            Unqualified name of the attribute to be added.\n        "
        self._global_attr_names.add(attr_name)

    def add_global_attrs_from_module(self, target_module):
        if False:
            i = 10
            return i + 15
        "\n        Record all global attributes (e.g., classes, variables) defined by the\n        target module corresponding to the passed graph node to also be defined\n        by the source module corresponding to this graph node.\n\n        If the source module is actually a package, this method instead records\n        these attributes to be defined by this package's pure-Python `__init__`\n        submodule.\n\n        Parameters\n        ----------\n        target_module : Node\n            Graph node of the target module to import attributes from.\n        "
        self._global_attr_names.update(target_module._global_attr_names)

    def add_submodule(self, submodule_basename, submodule_node):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add the submodule with the passed name and previously imported graph\n        node to the parent module corresponding to this graph node.\n\n        This parent module is typically but _not_ always a package (e.g., the\n        non-package `os` module containing the `os.path` submodule).\n\n        Parameters\n        ----------\n        submodule_basename : str\n            Unqualified name of the submodule to add to this parent module.\n        submodule_node : Node\n            Graph node of this submodule.\n        '
        self._submodule_basename_to_node[submodule_basename] = submodule_node

    def get_submodule(self, submodule_basename):
        if False:
            i = 10
            return i + 15
        '\n        Graph node of the submodule with the passed name in the parent module\n        corresponding to this graph node.\n\n        If this parent module does _not_ contain this submodule, an exception\n        is raised. Else, this parent module is typically but _not_ always a\n        package (e.g., the non-package `os` module containing the `os.path`\n        submodule).\n\n        Parameters\n        ----------\n        module_basename : str\n            Unqualified name of the submodule to retrieve.\n\n        Returns\n        ----------\n        Node\n            Graph node of this submodule.\n        '
        return self._submodule_basename_to_node[submodule_basename]

    def get_submodule_or_none(self, submodule_basename):
        if False:
            while True:
                i = 10
        '\n        Graph node of the submodule with the passed unqualified name in the\n        parent module corresponding to this graph node if this module contains\n        this submodule _or_ `None`.\n\n        This parent module is typically but _not_ always a package (e.g., the\n        non-package `os` module containing the `os.path` submodule).\n\n        Parameters\n        ----------\n        submodule_basename : str\n            Unqualified name of the submodule to retrieve.\n\n        Returns\n        ----------\n        Node\n            Graph node of this submodule if this parent module contains this\n            submodule _or_ `None`.\n        '
        return self._submodule_basename_to_node.get(submodule_basename)

    def remove_global_attr_if_found(self, attr_name):
        if False:
            print('Hello World!')
        '\n        Record the global attribute (e.g., class, variable) with the passed\n        name if previously recorded as defined by the pure-Python module\n        corresponding to this graph node to be subsequently undefined by the\n        same module.\n\n        If this module is actually a package, this method instead records this\n        attribute to be undefined by this package\'s pure-Python `__init__`\n        submodule.\n\n        This method is intended to be called on globals previously defined by\n        this module that are subsequently undefined via the `del` built-in by\n        this module, thus "forgetting" or "undoing" these globals.\n\n        For safety, there exists no corresponding `remove_global_attr()`\n        method. While defining this method is trivial, doing so would invite\n        `KeyError` exceptions on scanning valid Python that lexically deletes a\n        global in a scope under this module\'s top level (e.g., in a function)\n        _before_ defining this global at this top level. Since `ModuleGraph`\n        cannot and should not (re)implement a full-blown Python interpreter,\n        ignoring out-of-order deletions is the only sane policy.\n\n        Parameters\n        ----------\n        attr_name : str\n            Unqualified name of the attribute to be removed.\n        '
        if self.is_global_attr(attr_name):
            self._global_attr_names.remove(attr_name)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        try:
            otherIdent = getattr(other, 'graphident')
        except AttributeError:
            return False
        return self.graphident == otherIdent

    def __ne__(self, other):
        if False:
            return 10
        try:
            otherIdent = getattr(other, 'graphident')
        except AttributeError:
            return True
        return self.graphident != otherIdent

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        try:
            otherIdent = getattr(other, 'graphident')
        except AttributeError:
            return NotImplemented
        return self.graphident < otherIdent

    def __le__(self, other):
        if False:
            print('Hello World!')
        try:
            otherIdent = getattr(other, 'graphident')
        except AttributeError:
            return NotImplemented
        return self.graphident <= otherIdent

    def __gt__(self, other):
        if False:
            i = 10
            return i + 15
        try:
            otherIdent = getattr(other, 'graphident')
        except AttributeError:
            return NotImplemented
        return self.graphident > otherIdent

    def __ge__(self, other):
        if False:
            return 10
        try:
            otherIdent = getattr(other, 'graphident')
        except AttributeError:
            return NotImplemented
        return self.graphident >= otherIdent

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.graphident)

    def infoTuple(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.identifier,)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s%r' % (type(self).__name__, self.infoTuple())

class Alias(str):
    """
    Placeholder aliasing an existing source module to a non-existent target
    module (i.e., the desired alias).

    For obscure reasons, this class subclasses `str`. Each instance of this
    class is the fully-qualified name of the existing source module being
    aliased. Unlike the related `AliasNode` class, instances of this class are
    _not_ actual nodes and hence _not_ added to the graph; they only facilitate
    communication between the `ModuleGraph.alias_module()` and
    `ModuleGraph.find_node()` methods.
    """

class AliasNode(Node):
    """
    Graph node representing the aliasing of an existing source module under a
    non-existent target module name (i.e., the desired alias).
    """

    def __init__(self, name, node):
        if False:
            print('Hello World!')
        '\n        Initialize this alias.\n\n        Parameters\n        ----------\n        name : str\n            Fully-qualified name of the non-existent target module to be\n            created (as an alias of the existing source module).\n        node : Node\n            Graph node of the existing source module being aliased.\n        '
        super(AliasNode, self).__init__(name)
        for attr_name in ('identifier', 'packagepath', '_global_attr_names', '_starimported_ignored_module_names', '_submodule_basename_to_node'):
            if hasattr(node, attr_name):
                setattr(self, attr_name, getattr(node, attr_name))

    def infoTuple(self):
        if False:
            i = 10
            return i + 15
        return (self.graphident, self.identifier)

class BadModule(Node):
    pass

class ExcludedModule(BadModule):
    pass

class MissingModule(BadModule):
    pass

class InvalidRelativeImport(BadModule):

    def __init__(self, relative_path, from_name):
        if False:
            while True:
                i = 10
        identifier = relative_path
        if relative_path.endswith('.'):
            identifier += from_name
        else:
            identifier += '.' + from_name
        super(InvalidRelativeImport, self).__init__(identifier)
        self.relative_path = relative_path
        self.from_name = from_name

    def infoTuple(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.relative_path, self.from_name)

class Script(Node):

    def __init__(self, filename):
        if False:
            print('Hello World!')
        super(Script, self).__init__(filename)
        self.filename = filename

    def infoTuple(self):
        if False:
            print('Hello World!')
        return (self.filename,)

class BaseModule(Node):

    def __init__(self, name, filename=None, path=None):
        if False:
            for i in range(10):
                print('nop')
        super(BaseModule, self).__init__(name)
        self.filename = filename
        self.packagepath = path

    def infoTuple(self):
        if False:
            print('Hello World!')
        return tuple(filter(None, (self.identifier, self.filename, self.packagepath)))

class BuiltinModule(BaseModule):
    pass

class SourceModule(BaseModule):
    pass

class InvalidSourceModule(SourceModule):
    pass

class CompiledModule(BaseModule):
    pass

class InvalidCompiledModule(BaseModule):
    pass

class Extension(BaseModule):
    pass

class Package(BaseModule):
    """
    Graph node representing a non-namespace package.
    """
    pass

class ExtensionPackage(Extension, Package):
    """
    Graph node representing a package where the __init__ module is an extension
    module.
    """
    pass

class NamespacePackage(Package):
    """
    Graph node representing a namespace package.
    """
    pass

class RuntimeModule(BaseModule):
    """
    Graph node representing a non-package Python module dynamically defined at
    runtime.

    Most modules are statically defined on-disk as standard Python files.
    Some modules, however, are dynamically defined in-memory at runtime
    (e.g., `gi.repository.Gst`, dynamically defined by the statically
    defined `gi.repository.__init__` module).

    This node represents such a runtime module. Since this is _not_ a package,
    all attempts to import submodules from this module in `from`-style import
    statements (e.g., the `queue` submodule in `from six.moves import queue`)
    will be silently ignored.

    To ensure that the parent package of this module if any is also imported
    and added to the graph, this node is typically added to the graph by
    calling the `ModuleGraph.add_module()` method.
    """
    pass

class RuntimePackage(Package):
    """
    Graph node representing a non-namespace Python package dynamically defined
    at runtime.

    Most packages are statically defined on-disk as standard subdirectories
    containing `__init__.py` files. Some packages, however, are dynamically
    defined in-memory at runtime (e.g., `six.moves`, dynamically defined by
    the statically defined `six` module).

    This node represents such a runtime package. All attributes imported from
    this package in `from`-style import statements that are submodules of this
    package (e.g., the `queue` submodule in `from six.moves import queue`) will
    be imported rather than ignored.

    To ensure that the parent package of this package if any is also imported
    and added to the graph, this node is typically added to the graph by
    calling the `ModuleGraph.add_module()` method.
    """
    pass

class FlatPackage(BaseModule):

    def __init__(self, *args, **kwds):
        if False:
            while True:
                i = 10
        warnings.warn('This class will be removed in a future version of modulegraph', DeprecationWarning)
        super(FlatPackage, *args, **kwds)

class ArchiveModule(BaseModule):

    def __init__(self, *args, **kwds):
        if False:
            print('Hello World!')
        warnings.warn('This class will be removed in a future version of modulegraph', DeprecationWarning)
        super(FlatPackage, *args, **kwds)
header = '<!DOCTYPE html>\n<html>\n  <head>\n    <meta charset="UTF-8">\n    <title>%(TITLE)s</title>\n    <style>\n      .node { padding: 0.5em 0 0.5em; border-top: thin grey dotted; }\n      .moduletype { font: smaller italic }\n      .node a { text-decoration: none; color: #006699; }\n      .node a:visited { text-decoration: none; color: #2f0099; }\n    </style>\n  </head>\n  <body>\n    <h1>%(TITLE)s</h1>'
entry = '\n<div class="node">\n  <a name="%(NAME)s"></a>\n  %(CONTENT)s\n</div>'
contpl = '<tt>%(NAME)s</tt> <span class="moduletype">%(TYPE)s</span>'
contpl_linked = '<a target="code" href="%(URL)s" type="text/plain"><tt>%(NAME)s</tt></a>\n<span class="moduletype">%(TYPE)s</span>'
imports = '  <div class="import">\n%(HEAD)s:\n  %(LINKS)s\n  </div>\n'
footer = '\n  </body>\n</html>'

def _ast_names(names):
    if False:
        while True:
            i = 10
    result = []
    for nm in names:
        if isinstance(nm, ast.alias):
            result.append(nm.name)
        else:
            result.append(nm)
    result = [r for r in result if r != '__main__']
    return result

def uniq(seq):
    if False:
        while True:
            i = 10
    'Remove duplicates from a list, preserving order'
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
if sys.version_info[0] == 2:
    DEFAULT_IMPORT_LEVEL = -1
else:
    DEFAULT_IMPORT_LEVEL = 0

class _Visitor(ast.NodeVisitor):

    def __init__(self, graph, module):
        if False:
            while True:
                i = 10
        self._graph = graph
        self._module = module
        self._level = DEFAULT_IMPORT_LEVEL
        self._in_if = [False]
        self._in_def = [False]
        self._in_tryexcept = [False]

    @property
    def in_if(self):
        if False:
            i = 10
            return i + 15
        return self._in_if[-1]

    @property
    def in_def(self):
        if False:
            for i in range(10):
                print('nop')
        return self._in_def[-1]

    @property
    def in_tryexcept(self):
        if False:
            for i in range(10):
                print('nop')
        return self._in_tryexcept[-1]

    def _collect_import(self, name, fromlist, level):
        if False:
            print('Hello World!')
        if sys.version_info[0] == 2:
            if name == '__future__' and 'absolute_import' in (fromlist or ()):
                self._level = 0
        have_star = False
        if fromlist is not None:
            fromlist = uniq(fromlist)
            if '*' in fromlist:
                fromlist.remove('*')
                have_star = True
        self._module._deferred_imports.append((have_star, (name, self._module, fromlist, level), {'edge_attr': DependencyInfo(conditional=self.in_if, tryexcept=self.in_tryexcept, function=self.in_def, fromlist=False)}))

    def visit_Import(self, node):
        if False:
            i = 10
            return i + 15
        for nm in _ast_names(node.names):
            self._collect_import(nm, None, self._level)

    def visit_ImportFrom(self, node):
        if False:
            while True:
                i = 10
        level = node.level if node.level != 0 else self._level
        self._collect_import(node.module or '', _ast_names(node.names), level)

    def visit_If(self, node):
        if False:
            i = 10
            return i + 15
        self._in_if.append(True)
        self.generic_visit(node)
        self._in_if.pop()

    def visit_FunctionDef(self, node):
        if False:
            for i in range(10):
                print('nop')
        self._in_def.append(True)
        self.generic_visit(node)
        self._in_def.pop()
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Try(self, node):
        if False:
            for i in range(10):
                print('nop')
        self._in_tryexcept.append(True)
        self.generic_visit(node)
        self._in_tryexcept.pop()

    def visit_TryExcept(self, node):
        if False:
            print('Hello World!')
        self._in_tryexcept.append(True)
        self.generic_visit(node)
        self._in_tryexcept.pop()

    def visit_Expression(self, node):
        if False:
            while True:
                i = 10
        pass
    visit_BoolOp = visit_Expression
    visit_BinOp = visit_Expression
    visit_UnaryOp = visit_Expression
    visit_Lambda = visit_Expression
    visit_IfExp = visit_Expression
    visit_Dict = visit_Expression
    visit_Set = visit_Expression
    visit_ListComp = visit_Expression
    visit_SetComp = visit_Expression
    visit_ListComp = visit_Expression
    visit_GeneratorExp = visit_Expression
    visit_Compare = visit_Expression
    visit_Yield = visit_Expression
    visit_YieldFrom = visit_Expression
    visit_Await = visit_Expression
    visit_Call = visit_Expression
    visit_Await = visit_Expression

class ModuleGraph(ObjectGraph):
    """
    Directed graph whose nodes represent modules and edges represent
    dependencies between these modules.
    """

    def createNode(self, cls, name, *args, **kw):
        if False:
            while True:
                i = 10
        m = self.find_node(name)
        if m is None:
            m = super(ModuleGraph, self).createNode(cls, name, *args, **kw)
        return m

    def __init__(self, path=None, excludes=(), replace_paths=(), implies=(), graph=None, debug=0):
        if False:
            print('Hello World!')
        super(ModuleGraph, self).__init__(graph=graph, debug=debug)
        if path is None:
            path = sys.path
        self.path = path
        self.lazynodes = {}
        self.lazynodes.update(dict(implies))
        for m in excludes:
            self.lazynodes[m] = None
        self.replace_paths = replace_paths
        self._package_path_map = _packagePathMap
        self._legacy_ns_packages = {}

    def scan_legacy_namespace_packages(self):
        if False:
            while True:
                i = 10
        '\n        Resolve extra package `__path__` entries for legacy setuptools-based\n        namespace packages, by reading `namespace_packages.txt` from dist\n        metadata.\n        '
        legacy_ns_packages = defaultdict(lambda : set())
        for dist in importlib_metadata.distributions():
            ns_packages = dist.read_text('namespace_packages.txt')
            if ns_packages is None:
                continue
            ns_packages = ns_packages.splitlines()
            dist_path = getattr(dist, '_path')
            if dist_path is None:
                continue
            for package_name in ns_packages:
                path = os.path.join(str(dist_path.parent), *package_name.split('.'))
                legacy_ns_packages[package_name].add(path)
        self._legacy_ns_packages = {package_name: list(paths) for (package_name, paths) in legacy_ns_packages.items()}

    def implyNodeReference(self, node, other, edge_data=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create a reference from the passed source node to the passed other node,\n        implying the former to depend upon the latter.\n\n        While the source node _must_ be an existing graph node, the target node\n        may be either an existing graph node _or_ a fully-qualified module name.\n        In the latter case, the module with that name and all parent packages of\n        that module will be imported _without_ raising exceptions and for each\n        newly imported module or package:\n\n        * A new graph node will be created for that module or package.\n        * A reference from the passed source node to that module or package will\n          be created.\n\n        This method allows dependencies between Python objects _not_ importable\n        with standard techniques (e.g., module aliases, C extensions).\n\n        Parameters\n        ----------\n        node : str\n            Graph node for this reference's source module or package.\n        other : {Node, str}\n            Either a graph node _or_ fully-qualified name for this reference's\n            target module or package.\n        "
        if isinstance(other, Node):
            self._updateReference(node, other, edge_data)
        else:
            if isinstance(other, tuple):
                raise ValueError(other)
            others = self._safe_import_hook(other, node, None)
            for other in others:
                self._updateReference(node, other, edge_data)

    def outgoing(self, fromnode):
        if False:
            for i in range(10):
                print('nop')
        '\n        Yield all nodes that `fromnode` dependes on (that is,\n        all modules that `fromnode` imports.\n        '
        node = self.find_node(fromnode)
        (out_edges, _) = self.get_edges(node)
        return out_edges
    getReferences = outgoing

    def incoming(self, tonode, collapse_missing_modules=True):
        if False:
            i = 10
            return i + 15
        node = self.find_node(tonode)
        (_, in_edges) = self.get_edges(node)
        if collapse_missing_modules:
            for n in in_edges:
                if isinstance(n, MissingModule):
                    for n in self.incoming(n, False):
                        yield n
                else:
                    yield n
        else:
            for n in in_edges:
                yield n
    getReferers = incoming

    def hasEdge(self, fromnode, tonode):
        if False:
            i = 10
            return i + 15
        " Return True iff there is an edge from 'fromnode' to 'tonode' "
        fromnode = self.find_node(fromnode)
        tonode = self.find_node(tonode)
        return self.graph.edge_by_node(fromnode, tonode) is not None

    def foldReferences(self, packagenode):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create edges to/from `packagenode` based on the edges to/from all\n        submodules of that package _and_ then hide the graph nodes\n        corresponding to those submodules.\n        '
        pkg = self.find_node(packagenode)
        for n in self.nodes():
            if not n.identifier.startswith(pkg.identifier + '.'):
                continue
            (iter_out, iter_inc) = self.get_edges(n)
            for other in iter_out:
                if other.identifier.startswith(pkg.identifier + '.'):
                    continue
                if not self.hasEdge(pkg, other):
                    self._updateReference(pkg, other, 'pkg-internal-import')
            for other in iter_inc:
                if other.identifier.startswith(pkg.identifier + '.'):
                    continue
                if not self.hasEdge(other, pkg):
                    self._updateReference(other, pkg, 'pkg-import')
            self.graph.hide_node(n)

    def _updateReference(self, fromnode, tonode, edge_data):
        if False:
            i = 10
            return i + 15
        try:
            ed = self.edgeData(fromnode, tonode)
        except (KeyError, GraphError):
            return self.add_edge(fromnode, tonode, edge_data)
        if not (isinstance(ed, DependencyInfo) and isinstance(edge_data, DependencyInfo)):
            self.updateEdgeData(fromnode, tonode, edge_data)
        else:
            self.updateEdgeData(fromnode, tonode, ed._merged(edge_data))

    def add_edge(self, fromnode, tonode, edge_data='direct'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a reference from fromnode to tonode\n        '
        return super(ModuleGraph, self).createReference(fromnode, tonode, edge_data=edge_data)
    createReference = add_edge

    def find_node(self, name, create_nspkg=True):
        if False:
            i = 10
            return i + 15
        '\n        Graph node uniquely identified by the passed fully-qualified module\n        name if this module has been added to the graph _or_ `None` otherwise.\n\n        If (in order):\n\n        . A namespace package with this identifier exists _and_ the passed\n          `create_nspkg` parameter is `True`, this package will be\n          instantiated and returned.\n        . A lazy node with this identifier and:\n          * No dependencies exists, this node will be instantiated and\n            returned.\n          * Dependencies exists, this node and all transitive dependencies of\n            this node be instantiated and this node returned.\n        . A non-lazy node with this identifier exists, this node will be\n          returned as is.\n\n        Parameters\n        ----------\n        name : str\n            Fully-qualified name of the module whose graph node is to be found.\n        create_nspkg : bool\n            Ignored.\n\n        Returns\n        ----------\n        Node\n            Graph node of this module if added to the graph _or_ `None`\n            otherwise.\n        '
        data = super(ModuleGraph, self).findNode(name)
        if data is not None:
            return data
        if name in self.lazynodes:
            deps = self.lazynodes.pop(name)
            if deps is None:
                m = self.createNode(ExcludedModule, name)
            elif isinstance(deps, Alias):
                other = self._safe_import_hook(deps, None, None).pop()
                m = self.createNode(AliasNode, name, other)
                self.implyNodeReference(m, other)
            else:
                m = self._safe_import_hook(name, None, None).pop()
                for dep in deps:
                    self.implyNodeReference(m, dep)
            return m
        return None
    findNode = find_node
    iter_graph = ObjectGraph.flatten

    def add_script(self, pathname, caller=None):
        if False:
            i = 10
            return i + 15
        '\n        Create a node by path (not module name).  It is expected to be a Python\n        source file, and will be scanned for dependencies.\n        '
        self.msg(2, 'run_script', pathname)
        pathname = os.path.realpath(pathname)
        m = self.find_node(pathname)
        if m is not None:
            return m
        if sys.version_info[0] != 2:
            with open(pathname, 'rb') as fp:
                encoding = util.guess_encoding(fp)
            with open(pathname, _READ_MODE, encoding=encoding) as fp:
                contents = fp.read() + '\n'
            if contents.startswith(BOM):
                contents = contents[1:]
        else:
            with open(pathname, _READ_MODE) as fp:
                contents = fp.read() + '\n'
        co_ast = compile(contents, pathname, 'exec', ast.PyCF_ONLY_AST, True)
        co = compile(co_ast, pathname, 'exec', 0, True)
        m = self.createNode(Script, pathname)
        self._updateReference(caller, m, None)
        n = self._scan_code(m, co, co_ast)
        self._process_imports(n)
        m.code = co
        if self.replace_paths:
            m.code = self._replace_paths_in_code(m.code)
        return m

    def import_hook(self, target_module_partname, source_module=None, target_attr_names=None, level=DEFAULT_IMPORT_LEVEL, edge_attr=None):
        if False:
            i = 10
            return i + 15
        '\n        Import the module with the passed name, all parent packages of this\n        module, _and_ all submodules and attributes in this module with the\n        passed names from the previously imported caller module signified by\n        the passed graph node.\n\n        Unlike most import methods (e.g., `_safe_import_hook()`), this method\n        is designed to be publicly called by both external and internal\n        callers and hence is public.\n\n        Parameters\n        ----------\n        target_module_partname : str\n            Partially-qualified name of the target module to be imported. See\n            `_safe_import_hook()` for further details.\n        source_module : Node\n            Graph node for the previously imported **source module** (i.e.,\n            module containing the `import` statement triggering the call to\n            this method) _or_ `None` if this module is to be imported in a\n            "disconnected" manner. **Passing `None` is _not_ recommended.**\n            Doing so produces a disconnected graph in which the graph node\n            created for the module to be imported will be disconnected and\n            hence unreachable from all other nodes -- which frequently causes\n            subtle issues in external callers (namely PyInstaller, which\n            silently ignores unreachable nodes).\n        target_attr_names : list\n            List of the unqualified names of all submodules and attributes to\n            be imported from the module to be imported if this is a "from"-\n            style import (e.g., `[encode_base64, encode_noop]` for the import\n            `from email.encoders import encode_base64, encode_noop`) _or_\n            `None` otherwise.\n        level : int\n            Whether to perform an absolute or relative import. See\n            `_safe_import_hook()` for further details.\n\n        Returns\n        ----------\n        list\n            List of the graph nodes created for all modules explicitly imported\n            by this call, including the passed module and all submodules listed\n            in `target_attr_names` _but_ excluding all parent packages\n            implicitly imported by this call. If `target_attr_names` is `None`\n            or the empty list, this is guaranteed to be a list of one element:\n            the graph node created for the passed module.\n\n        Raises\n        ----------\n        ImportError\n            If the target module to be imported is unimportable.\n        '
        self.msg(3, '_import_hook', target_module_partname, source_module, source_module, level)
        source_package = self._determine_parent(source_module)
        (target_package, target_module_partname) = self._find_head_package(source_package, target_module_partname, level)
        self.msgin(4, 'load_tail', target_package, target_module_partname)
        submodule = target_package
        while target_module_partname:
            i = target_module_partname.find('.')
            if i < 0:
                i = len(target_module_partname)
            (head, target_module_partname) = (target_module_partname[:i], target_module_partname[i + 1:])
            mname = '%s.%s' % (submodule.identifier, head)
            submodule = self._safe_import_module(head, mname, submodule)
            if submodule is None:
                self.msgout(4, 'raise ImportError: No module named', mname)
                raise ImportError('No module named ' + repr(mname))
        self.msgout(4, 'load_tail ->', submodule)
        target_module = submodule
        target_modules = [target_module]
        if target_attr_names and isinstance(target_module, (Package, AliasNode)):
            for target_submodule in self._import_importable_package_submodules(target_module, target_attr_names):
                if target_submodule not in target_modules:
                    target_modules.append(target_submodule)
        for target_module in target_modules:
            self._updateReference(source_module, target_module, edge_data=edge_attr)
        return target_modules

    def _determine_parent(self, caller):
        if False:
            print('Hello World!')
        '\n        Determine the package containing a node.\n        '
        self.msgin(4, 'determine_parent', caller)
        parent = None
        if caller:
            pname = caller.identifier
            if isinstance(caller, Package):
                parent = caller
            elif '.' in pname:
                pname = pname[:pname.rfind('.')]
                parent = self.find_node(pname)
            elif caller.packagepath:
                parent = self.find_node(pname)
        self.msgout(4, 'determine_parent ->', parent)
        return parent

    def _find_head_package(self, source_package, target_module_partname, level=DEFAULT_IMPORT_LEVEL):
        if False:
            return 10
        '\n        Import the target package providing the target module with the passed\n        name to be subsequently imported from the previously imported source\n        package corresponding to the passed graph node.\n\n        Parameters\n        ----------\n        source_package : Package\n            Graph node for the previously imported **source package** (i.e.,\n            package containing the module containing the `import` statement\n            triggering the call to this method) _or_ `None` if this module is\n            to be imported in a "disconnected" manner. **Passing `None` is\n            _not_ recommended.** See the `_import_hook()` method for further\n            details.\n        target_module_partname : str\n            Partially-qualified name of the target module to be imported. See\n            `_safe_import_hook()` for further details.\n        level : int\n            Whether to perform absolute or relative imports. See the\n            `_safe_import_hook()` method for further details.\n\n        Returns\n        ----------\n        (target_package, target_module_tailname)\n            2-tuple describing the imported target package, where:\n            * `target_package` is the graph node created for this package.\n            * `target_module_tailname` is the unqualified name of the target\n              module to be subsequently imported (e.g., `text` when passed a\n              `target_module_partname` of `email.mime.text`).\n\n        Raises\n        ----------\n        ImportError\n            If the package to be imported is unimportable.\n        '
        self.msgin(4, 'find_head_package', source_package, target_module_partname, level)
        if '.' in target_module_partname:
            (target_module_headname, target_module_tailname) = target_module_partname.split('.', 1)
        else:
            target_module_headname = target_module_partname
            target_module_tailname = ''
        if level == ABSOLUTE_OR_RELATIVE_IMPORT_LEVEL:
            if source_package:
                target_package_name = source_package.identifier + '.' + target_module_headname
            else:
                target_package_name = target_module_headname
        elif level == ABSOLUTE_IMPORT_LEVEL:
            target_package_name = target_module_headname
            source_package = None
        else:
            if source_package is None:
                self.msg(2, 'Relative import outside of package')
                raise InvalidRelativeImportError('Relative import outside of package (name=%r, parent=%r, level=%r)' % (target_module_partname, source_package, level))
            for i in range(level - 1):
                if '.' not in source_package.identifier:
                    self.msg(2, 'Relative import outside of package')
                    raise InvalidRelativeImportError('Relative import outside of package (name=%r, parent=%r, level=%r)' % (target_module_partname, source_package, level))
                p_fqdn = source_package.identifier.rsplit('.', 1)[0]
                new_parent = self.find_node(p_fqdn)
                if new_parent is None:
                    self.msg(2, 'Relative import outside of package')
                    raise InvalidRelativeImportError('Relative import outside of package (name=%r, parent=%r, level=%r)' % (target_module_partname, source_package, level))
                assert new_parent is not source_package, (new_parent, source_package)
                source_package = new_parent
            if target_module_headname:
                target_package_name = source_package.identifier + '.' + target_module_headname
            else:
                target_package_name = source_package.identifier
        target_package = self._safe_import_module(target_module_headname, target_package_name, source_package)
        if target_package is None and source_package is not None and (level <= ABSOLUTE_IMPORT_LEVEL):
            target_package_name = target_module_headname
            source_package = None
            target_package = self._safe_import_module(target_module_headname, target_package_name, source_package)
        if target_package is not None:
            self.msgout(4, 'find_head_package ->', (target_package, target_module_tailname))
            return (target_package, target_module_tailname)
        self.msgout(4, 'raise ImportError: No module named', target_package_name)
        raise ImportError('No module named ' + target_package_name)

    def _import_importable_package_submodules(self, package, attr_names):
        if False:
            i = 10
            return i + 15
        '\n        Generator importing and yielding each importable submodule (of the\n        previously imported package corresponding to the passed graph node)\n        whose unqualified name is in the passed list.\n\n        Elements of this list that are _not_ importable submodules of this\n        package are either:\n\n        * Ignorable attributes (e.g., classes, globals) defined at the top\n          level of this package\'s `__init__` submodule, which will be ignored.\n        * Else, unignorable unimportable submodules, in which case an\n          exception is raised.\n\n        Parameters\n        ----------\n        package : Package\n            Graph node of the previously imported package containing the\n            modules to be imported and yielded.\n\n        attr_names : list\n            List of the unqualified names of all attributes of this package to\n            attempt to import as submodules. This list will be internally\n            converted into a set, safely ignoring any duplicates in this list\n            (e.g., reducing the "from"-style import\n            `from foo import bar, car, far, bar, car, far` to merely\n            `from foo import bar, car, far`).\n\n        Yields\n        ----------\n        Node\n            Graph node created for the currently imported submodule.\n\n        Raises\n        ----------\n        ImportError\n            If any attribute whose name is in `attr_names` is neither:\n            * An importable submodule of this package.\n            * An ignorable global attribute (e.g., class, variable) defined at\n              the top level of this package\'s `__init__` submodule.\n            In this case, this attribute _must_ be an unimportable submodule of\n            this package.\n        '
        attr_names = set(attr_names)
        self.msgin(4, '_import_importable_package_submodules', package, attr_names)
        if '*' in attr_names:
            attr_names.update(self._find_all_submodules(package))
            attr_names.remove('*')
        for attr_name in attr_names:
            submodule = package.get_submodule_or_none(attr_name)
            if submodule is None:
                submodule_name = package.identifier + '.' + attr_name
                submodule = self._safe_import_module(attr_name, submodule_name, package)
                if submodule is None:
                    if package.is_global_attr(attr_name):
                        self.msg(4, '_import_importable_package_submodules: ignoring from-imported global', package.identifier, attr_name)
                        continue
                    else:
                        raise ImportError('No module named ' + submodule_name)
            yield submodule
        self.msgin(4, '_import_importable_package_submodules ->')

    def _find_all_submodules(self, m):
        if False:
            i = 10
            return i + 15
        if not m.packagepath:
            return
        for path in m.packagepath:
            try:
                names = zipio.listdir(path)
            except (os.error, IOError):
                self.msg(2, "can't list directory", path)
                continue
            for name in names:
                for suffix in importlib.machinery.all_suffixes():
                    if path.endswith(suffix):
                        name = os.path.basename(path)[:-len(suffix)]
                        break
                else:
                    continue
                if name != '__init__':
                    yield name

    def alias_module(self, src_module_name, trg_module_name):
        if False:
            while True:
                i = 10
        '\n        Alias the source module to the target module with the passed names.\n\n        This method ensures that the next call to findNode() given the target\n        module name will resolve this alias. This includes importing and adding\n        a graph node for the source module if needed as well as adding a\n        reference from the target to source module.\n\n        Parameters\n        ----------\n        src_module_name : str\n            Fully-qualified name of the existing **source module** (i.e., the\n            module being aliased).\n        trg_module_name : str\n            Fully-qualified name of the non-existent **target module** (i.e.,\n            the alias to be created).\n        '
        self.msg(3, 'alias_module "%s" -> "%s"' % (src_module_name, trg_module_name))
        assert isinstance(src_module_name, str), '"%s" not a module name.' % str(src_module_name)
        assert isinstance(trg_module_name, str), '"%s" not a module name.' % str(trg_module_name)
        trg_module = self.find_node(trg_module_name)
        if trg_module is not None and (not (isinstance(trg_module, AliasNode) and trg_module.identifier == src_module_name)):
            raise ValueError('Target module "%s" already imported as "%s".' % (trg_module_name, trg_module))
        self.lazynodes[trg_module_name] = Alias(src_module_name)

    def add_module(self, module):
        if False:
            while True:
                i = 10
        "\n        Add the passed module node to the graph if not already added.\n\n        If that module has a parent module or package with a previously added\n        node, this method also adds a reference from this module node to its\n        parent node and adds this module node to its parent node's namespace.\n\n        This high-level method wraps the low-level `addNode()` method, but is\n        typically _only_ called by graph hooks adding runtime module nodes. For\n        all other node types, the `import_module()` method should be called.\n\n        Parameters\n        ----------\n        module : BaseModule\n            Graph node of the module to be added.\n        "
        self.msg(3, 'add_module', module)
        module_added = self.find_node(module.identifier)
        if module_added is None:
            self.addNode(module)
        else:
            assert module == module_added, 'New module %r != previous %r.' % (module, module_added)
        (parent_name, _, module_basename) = module.identifier.rpartition('.')
        if parent_name:
            parent = self.find_node(parent_name)
            if parent is None:
                self.msg(4, 'add_module parent not found:', parent_name)
            else:
                self.add_edge(module, parent)
                parent.add_submodule(module_basename, module)

    def append_package_path(self, package_name, directory):
        if False:
            for i in range(10):
                print('nop')
        "\n        Modulegraph does a good job at simulating Python's, but it can not\n        handle packagepath '__path__' modifications packages make at runtime.\n\n        Therefore there is a mechanism whereby you can register extra paths\n        in this map for a package, and it will be honored.\n\n        NOTE: This method has to be called before a package is resolved by\n              modulegraph.\n\n        Parameters\n        ----------\n        module : str\n            Fully-qualified module name.\n        directory : str\n            Absolute or relative path of the directory to append to the\n            '__path__' attribute.\n        "
        paths = self._package_path_map.setdefault(package_name, [])
        paths.append(directory)

    def _safe_import_module(self, module_partname, module_name, parent_module):
        if False:
            return 10
        "\n        Create a new graph node for the module with the passed name under the\n        parent package signified by the passed graph node _without_ raising\n        `ImportError` exceptions.\n\n        If this module has already been imported, this module's existing graph\n        node will be returned; else if this module is importable, a new graph\n        node will be added for this module and returned; else this module is\n        unimportable, in which case `None` will be returned. Like the\n        `_safe_import_hook()` method, this method does _not_ raise\n        `ImportError` exceptions when this module is unimportable.\n\n        Parameters\n        ----------\n        module_partname : str\n            Unqualified name of the module to be imported (e.g., `text`).\n        module_name : str\n            Fully-qualified name of this module (e.g., `email.mime.text`).\n        parent_module : Package\n            Graph node of the previously imported parent module containing this\n            submodule _or_ `None` if this is a top-level module (i.e.,\n            `module_name` contains no `.` delimiters). This parent module is\n            typically but _not_ always a package (e.g., the `os.path` submodule\n            contained by the `os` module).\n\n        Returns\n        ----------\n        Node\n            Graph node created for this module _or_ `None` if this module is\n            unimportable.\n        "
        self.msgin(3, 'safe_import_module', module_partname, module_name, parent_module)
        module = self.find_node(module_name)
        if module is None:
            search_dirs = None
            if parent_module is not None:
                if parent_module.packagepath is not None:
                    search_dirs = parent_module.packagepath
                else:
                    self.msgout(3, 'safe_import_module -> None (parent_parent.packagepath is None)')
                    return None
            try:
                (pathname, loader) = self._find_module(module_partname, search_dirs, parent_module)
            except ImportError as exc:
                self.msgout(3, 'safe_import_module -> None (%r)' % exc)
                return None
            (module, co) = self._load_module(module_name, pathname, loader)
            if co is not None:
                try:
                    if isinstance(co, ast.AST):
                        co_ast = co
                        co = compile(co_ast, pathname, 'exec', 0, True)
                    else:
                        co_ast = None
                    n = self._scan_code(module, co, co_ast)
                    self._process_imports(n)
                    if self.replace_paths:
                        co = self._replace_paths_in_code(co)
                    module.code = co
                except SyntaxError:
                    self.msg(1, 'safe_import_module: SyntaxError in ', pathname)
                    cls = InvalidSourceModule
                    module = self.createNode(cls, module_name)
        if parent_module is not None:
            self.msg(4, 'safe_import_module create reference', module, '->', parent_module)
            self._updateReference(module, parent_module, edge_data=DependencyInfo(conditional=False, fromlist=False, function=False, tryexcept=False))
            parent_module.add_submodule(module_partname, module)
        self.msgout(3, 'safe_import_module ->', module)
        return module

    def _load_module(self, fqname, pathname, loader):
        if False:
            i = 10
            return i + 15
        from importlib._bootstrap_external import ExtensionFileLoader
        self.msgin(2, 'load_module', fqname, pathname, loader.__class__.__name__)
        partname = fqname.rpartition('.')[-1]
        if loader.is_package(partname):
            if isinstance(loader, NAMESPACE_PACKAGE):
                m = self.createNode(NamespacePackage, fqname)
                m.filename = '-'
                m.packagepath = loader.namespace_dirs[:]
            else:
                ns_pkgpaths = self._legacy_ns_packages.get(fqname, [])
                if isinstance(loader, ExtensionFileLoader):
                    m = self.createNode(ExtensionPackage, fqname)
                else:
                    m = self.createNode(Package, fqname)
                m.filename = pathname
                assert os.path.basename(pathname).startswith('__init__.')
                m.packagepath = [os.path.dirname(pathname)] + ns_pkgpaths
            m.packagepath = m.packagepath + self._package_path_map.get(fqname, [])
            if isinstance(m, NamespacePackage):
                return (m, None)
        co = None
        if loader is BUILTIN_MODULE:
            cls = BuiltinModule
        elif isinstance(loader, ExtensionFileLoader):
            cls = Extension
        else:
            try:
                src = loader.get_source(partname)
            except (UnicodeDecodeError, SyntaxError) as e:
                if isinstance(e, SyntaxError):
                    if not isinstance(e.__context__, UnicodeDecodeError):
                        raise
                self.msg(2, f'load_module: failed to obtain source for {partname}: {e}! Falling back to reading as raw data!')
                path = loader.get_filename(partname)
                src = loader.get_data(path)
            if src is not None:
                try:
                    co = compile(src, pathname, 'exec', ast.PyCF_ONLY_AST, True)
                    cls = SourceModule
                    if sys.version_info[:2] == (3, 5):
                        compile(co, '-', 'exec', 0, True)
                except SyntaxError:
                    co = None
                    cls = InvalidSourceModule
                except Exception as exc:
                    cls = InvalidSourceModule
                    self.msg(2, 'load_module: InvalidSourceModule', pathname, exc)
            else:
                try:
                    co = loader.get_code(partname)
                    cls = CompiledModule if co is not None else InvalidCompiledModule
                except Exception as exc:
                    self.msg(2, 'load_module: InvalidCompiledModule, Cannot load code', pathname, exc)
                    cls = InvalidCompiledModule
        m = self.createNode(cls, fqname)
        m.filename = pathname
        self.msgout(2, 'load_module ->', m)
        return (m, co)

    def _safe_import_hook(self, target_module_partname, source_module, target_attr_names, level=DEFAULT_IMPORT_LEVEL, edge_attr=None):
        if False:
            while True:
                i = 10
        '\n        Import the module with the passed name and all parent packages of this\n        module from the previously imported caller module signified by the\n        passed graph node _without_ raising `ImportError` exceptions.\n\n        This method wraps the lowel-level `_import_hook()` method. On catching\n        an `ImportError` exception raised by that method, this method creates\n        and adds a `MissingNode` instance describing the unimportable module to\n        the graph instead.\n\n        Parameters\n        ----------\n        target_module_partname : str\n            Partially-qualified name of the module to be imported. If `level`\n            is:\n            * `ABSOLUTE_OR_RELATIVE_IMPORT_LEVEL` (e.g., the Python 2 default)\n              or a positive integer (e.g., an explicit relative import), the\n              fully-qualified name of this module is the concatenation of the\n              fully-qualified name of the caller module\'s package and this\n              parameter.\n            * `ABSOLUTE_IMPORT_LEVEL` (e.g., the Python 3 default), this name\n              is already fully-qualified.\n            * A non-negative integer (e.g., `1`), this name is typically the\n              empty string. In this case, this is a "from"-style relative\n              import (e.g., "from . import bar") and the fully-qualified name\n              of this module is dynamically resolved by import machinery.\n        source_module : Node\n            Graph node for the previously imported **caller module** (i.e.,\n            module containing the `import` statement triggering the call to\n            this method) _or_ `None` if this module is to be imported in a\n            "disconnected" manner. **Passing `None` is _not_ recommended.**\n            Doing so produces a disconnected graph in which the graph node\n            created for the module to be imported will be disconnected and\n            hence unreachable from all other nodes -- which frequently causes\n            subtle issues in external callers (e.g., PyInstaller, which\n            silently ignores unreachable nodes).\n        target_attr_names : list\n            List of the unqualified names of all submodules and attributes to\n            be imported via a `from`-style import statement from this target\n            module if any (e.g., the list `[encode_base64, encode_noop]` for\n            the import `from email.encoders import encode_base64, encode_noop`)\n            _or_ `None` otherwise. Ignored unless `source_module` is the graph\n            node of a package (i.e., is an instance of the `Package` class).\n            Why? Because:\n            * Consistency. The `_import_importable_package_submodules()`\n              method accepts a similar list applicable only to packages.\n            * Efficiency. Unlike packages, modules cannot physically contain\n              submodules. Hence, any target module imported via a `from`-style\n              import statement as an attribute from another target parent\n              module must itself have been imported in that target parent\n              module. The import statement responsible for that import must\n              already have been previously parsed by `ModuleGraph`, in which\n              case that target module will already be frozen by PyInstaller.\n              These imports are safely ignorable here.\n        level : int\n            Whether to perform an absolute or relative import. This parameter\n            corresponds exactly to the parameter of the same name accepted by\n            the `__import__()` built-in: "The default is -1 which indicates\n            both absolute and relative imports will be attempted. 0 means only\n            perform absolute imports. Positive values for level indicate the\n            number of parent directories to search relative to the directory of\n            the module calling `__import__()`." Defaults to -1 under Python 2\n            and 0 under Python 3. Since this default depends on the major\n            version of the current Python interpreter, depending on this\n            default can result in unpredictable and non-portable behaviour.\n            Callers are strongly recommended to explicitly pass this parameter\n            rather than implicitly accept this default.\n\n        Returns\n        ----------\n        list\n            List of the graph nodes created for all modules explicitly imported\n            by this call, including the passed module and all submodules listed\n            in `target_attr_names` _but_ excluding all parent packages\n            implicitly imported by this call. If `target_attr_names` is either\n            `None` or the empty list, this is guaranteed to be a list of one\n            element: the graph node created for the passed module. As above,\n            `MissingNode` instances are created for all unimportable modules.\n        '
        self.msg(3, '_safe_import_hook', target_module_partname, source_module, target_attr_names, level)

        def is_swig_candidate():
            if False:
                return 10
            return source_module is not None and target_attr_names is None and (level == ABSOLUTE_IMPORT_LEVEL) and (type(source_module) is SourceModule) and (target_module_partname == '_' + source_module.identifier.rpartition('.')[2]) and (sys.version_info[0] == 3)

        def is_swig_wrapper(source_module):
            if False:
                while True:
                    i = 10
            with open(source_module.filename, 'rb') as source_module_file:
                encoding = util.guess_encoding(source_module_file)
            with open(source_module.filename, _READ_MODE, encoding=encoding) as source_module_file:
                first_line = source_module_file.readline()
            self.msg(5, 'SWIG wrapper candidate first line: %r' % first_line)
            return 'automatically generated by SWIG' in first_line
        target_modules = None
        is_swig_import = None
        try:
            target_modules = self.import_hook(target_module_partname, source_module, target_attr_names=None, level=level, edge_attr=edge_attr)
        except InvalidRelativeImportError:
            self.msgout(2, 'Invalid relative import', level, target_module_partname, target_attr_names)
            result = []
            for sub in target_attr_names or '*':
                m = self.createNode(InvalidRelativeImport, '.' * level + target_module_partname, sub)
                self._updateReference(source_module, m, edge_data=edge_attr)
                result.append(m)
            return result
        except ImportError as msg:
            if is_swig_candidate():
                self.msg(4, 'SWIG import candidate (name=%r, caller=%r, level=%r)' % (target_module_partname, source_module, level))
                is_swig_import = is_swig_wrapper(source_module)
                if is_swig_import:
                    target_attr_names = [target_module_partname]
                    target_module_partname = ''
                    level = 1
                    self.msg(2, 'SWIG import (caller=%r, fromlist=%r, level=%r)' % (source_module, target_attr_names, level))
                    try:
                        target_modules = self.import_hook(target_module_partname, source_module, target_attr_names=None, level=level, edge_attr=edge_attr)
                    except ImportError as msg:
                        self.msg(2, 'SWIG ImportError:', str(msg))
            if target_modules is None:
                self.msg(2, 'ImportError:', str(msg))
                target_module = self.createNode(MissingModule, _path_from_importerror(msg, target_module_partname))
                self._updateReference(source_module, target_module, edge_data=edge_attr)
                target_modules = [target_module]
        assert len(target_modules) == 1, 'Expected import_hook() toreturn only one module but received: {}'.format(target_modules)
        target_module = target_modules[0]
        if isinstance(target_module, MissingModule) and is_swig_import is None and is_swig_candidate() and is_swig_wrapper(source_module):
            self.removeNode(target_module)
            target_modules = self._safe_import_hook(target_module_partname, source_module, target_attr_names=None, level=1, edge_attr=edge_attr)
            return target_modules
        if isinstance(edge_attr, DependencyInfo):
            edge_attr = edge_attr._replace(fromlist=True)
        if target_attr_names and isinstance(target_module, (Package, AliasNode)):
            for target_submodule_partname in target_attr_names:
                if target_module.is_submodule(target_submodule_partname):
                    target_submodule = target_module.get_submodule(target_submodule_partname)
                    if target_submodule is not None:
                        if target_submodule not in target_modules:
                            self._updateReference(source_module, target_submodule, edge_data=edge_attr)
                            target_modules.append(target_submodule)
                        continue
                target_submodule_name = target_module.identifier + '.' + target_submodule_partname
                target_submodule = self.find_node(target_submodule_name)
                if target_submodule is None:
                    try:
                        self.import_hook(target_module_partname, source_module, target_attr_names=[target_submodule_partname], level=level, edge_attr=edge_attr)
                        target_submodule = self.find_node(target_submodule_name)
                        if target_submodule is None:
                            assert target_module.is_global_attr(target_submodule_partname), 'No global named {} in {}.__init__'.format(target_submodule_partname, target_module.identifier)
                            self.msg(4, '_safe_import_hook', 'ignoring imported non-module global', target_module.identifier, target_submodule_partname)
                            continue
                        if is_swig_import:
                            if self.find_node(target_submodule_partname):
                                self.msg(2, 'SWIG import error: %r basename %r already exists' % (target_submodule_name, target_submodule_partname))
                            else:
                                self.msg(4, 'SWIG import renamed from %r to %r' % (target_submodule_name, target_submodule_partname))
                                target_submodule.identifier = target_submodule_partname
                    except ImportError as msg:
                        self.msg(2, 'ImportError:', str(msg))
                        target_submodule = self.createNode(MissingModule, target_submodule_name)
                target_module.add_submodule(target_submodule_partname, target_submodule)
                if target_submodule is not None:
                    self._updateReference(target_module, target_submodule, edge_data=edge_attr)
                    self._updateReference(source_module, target_submodule, edge_data=edge_attr)
                    if target_submodule not in target_modules:
                        target_modules.append(target_submodule)
        return target_modules

    def _scan_code(self, module, module_code_object, module_code_object_ast=None):
        if False:
            return 10
        "\n        Parse and add all import statements from the passed code object of the\n        passed source module to this graph, recursively.\n\n        **This method is at the root of all `ModuleGraph` recursion.**\n        Recursion begins here and ends when all import statements in all code\n        objects of all modules transitively imported by the source module\n        passed to the first call to this method have been added to the graph.\n        Specifically, this method:\n\n        1. If the passed `module_code_object_ast` parameter is non-`None`,\n           parses all import statements from this object.\n        2. Else, parses all import statements from the passed\n           `module_code_object` parameter.\n        1. For each such import statement:\n           1. Adds to this `ModuleGraph` instance:\n              1. Nodes for all target modules of these imports.\n              1. Directed edges from this source module to these target\n                 modules.\n           2. Recursively calls this method with these target modules.\n\n        Parameters\n        ----------\n        module : Node\n            Graph node of the module to be parsed.\n        module_code_object : PyCodeObject\n            Code object providing this module's disassembled Python bytecode.\n            Ignored unless `module_code_object_ast` is `None`.\n        module_code_object_ast : optional[ast.AST]\n            Optional abstract syntax tree (AST) of this module if any or `None`\n            otherwise. Defaults to `None`, in which case the passed\n            `module_code_object` is parsed instead.\n        Returns\n        ----------\n        module : Node\n            Graph node of the module to be parsed.\n        "
        module._deferred_imports = []
        if module_code_object_ast is not None:
            self._scan_ast(module, module_code_object_ast)
            self._scan_bytecode(module, module_code_object, is_scanning_imports=False)
        else:
            self._scan_bytecode(module, module_code_object, is_scanning_imports=True)
        return module

    def _scan_ast(self, module, module_code_object_ast):
        if False:
            print('Hello World!')
        '\n        Parse and add all import statements from the passed abstract syntax\n        tree (AST) of the passed source module to this graph, non-recursively.\n\n        Parameters\n        ----------\n        module : Node\n            Graph node of the module to be parsed.\n        module_code_object_ast : ast.AST\n            Abstract syntax tree (AST) of this module to be parsed.\n        '
        visitor = _Visitor(self, module)
        visitor.visit(module_code_object_ast)

    def _scan_bytecode(self, module, module_code_object, is_scanning_imports):
        if False:
            return 10
        "\n        Parse and add all import statements from the passed code object of the\n        passed source module to this graph, non-recursively.\n\n        This method parses all reasonably parsable operations (i.e., operations\n        that are both syntactically and semantically parsable _without_\n        requiring Turing-complete interpretation) directly or indirectly\n        involving module importation from this code object. This includes:\n\n        * `IMPORT_NAME`, denoting an import statement. Ignored unless\n          the passed `is_scanning_imports` parameter is `True`.\n        * `STORE_NAME` and `STORE_GLOBAL`, denoting the\n          declaration of a global attribute (e.g., class, variable) in this\n          module. This method stores each such declaration for subsequent\n          lookup. While global attributes are usually irrelevant to import\n          parsing, they remain the only means of distinguishing erroneous\n          non-ignorable attempts to import non-existent submodules of a package\n          from successful ignorable attempts to import existing global\n          attributes of a package's `__init__` submodule (e.g., the `bar` in\n          `from foo import bar`, which is either a non-ignorable submodule of\n          `foo` or an ignorable global attribute of `foo.__init__`).\n        * `DELETE_NAME` and `DELETE_GLOBAL`, denoting the\n          undeclaration of a previously declared global attribute in this\n          module.\n\n        Since `ModuleGraph` is _not_ intended to replicate the behaviour of a\n        full-featured Turing-complete Python interpreter, this method ignores\n        operations that are _not_ reasonably parsable from this code object --\n        even those directly or indirectly involving module importation. This\n        includes:\n\n        * `STORE_ATTR(namei)`, implementing `TOS.name = TOS1`. If `TOS` is the\n          name of a target module currently imported into the namespace of the\n          passed source module, this opcode would ideally be parsed to add that\n          global attribute to that target module. Since this addition only\n          conditionally occurs on the importation of this source module and\n          execution of the code branch in this module performing this addition,\n          however, that global _cannot_ be unconditionally added to that target\n          module. In short, only Turing-complete behaviour suffices.\n        * `DELETE_ATTR(namei)`, implementing `del TOS.name`. If `TOS` is the\n          name of a target module currently imported into the namespace of the\n          passed source module, this opcode would ideally be parsed to remove\n          that global attribute from that target module. Again, however, only\n          Turing-complete behaviour suffices.\n\n        Parameters\n        ----------\n        module : Node\n            Graph node of the module to be parsed.\n        module_code_object : PyCodeObject\n            Code object of the module to be parsed.\n        is_scanning_imports : bool\n            `True` only if this method is parsing import statements from\n            `IMPORT_NAME` opcodes. If `False`, no import statements will be\n            parsed. This parameter is typically:\n            * `True` when parsing this module's code object for such imports.\n            * `False` when parsing this module's abstract syntax tree (AST)\n              (rather than code object) for such imports. In this case, that\n              parsing will have already parsed import statements, which this\n              parsing must avoid repeating.\n        "
        level = None
        fromlist = None
        prev_insts = deque(maxlen=2)
        for inst in util.iterate_instructions(module_code_object):
            if not inst:
                continue
            if inst.opname == 'IMPORT_NAME':
                if not is_scanning_imports:
                    continue
                assert prev_insts[-2].opname == 'LOAD_CONST'
                assert prev_insts[-1].opname == 'LOAD_CONST'
                level = prev_insts[-2].argval
                fromlist = prev_insts[-1].argval
                assert fromlist is None or type(fromlist) is tuple
                target_module_partname = inst.argval
                have_star = False
                if fromlist is not None:
                    fromlist = uniq(fromlist)
                    if '*' in fromlist:
                        fromlist.remove('*')
                        have_star = True
                module._deferred_imports.append((have_star, (target_module_partname, module, fromlist, level), {}))
            elif inst.opname in ('STORE_NAME', 'STORE_GLOBAL'):
                name = inst.argval
                module.add_global_attr(name)
            elif inst.opname in ('DELETE_NAME', 'DELETE_GLOBAL'):
                name = inst.argval
                module.remove_global_attr_if_found(name)
            prev_insts.append(inst)

    def _process_imports(self, source_module):
        if False:
            return 10
        '\n        Graph all target modules whose importations were previously parsed from\n        the passed source module by a prior call to the `_scan_code()` method\n        and methods call by that method (e.g., `_scan_ast()`,\n        `_scan_bytecode()`, `_scan_bytecode_stores()`).\n\n        Parameters\n        ----------\n        source_module : Node\n            Graph node of the source module to graph target imports for.\n        '
        if not source_module._deferred_imports:
            return
        for (have_star, import_info, kwargs) in source_module._deferred_imports:
            target_modules = self._safe_import_hook(*import_info, **kwargs)
            if not target_modules:
                continue
            target_module = target_modules[0]
            if have_star:
                source_module.add_global_attrs_from_module(target_module)
                source_module._starimported_ignored_module_names.update(target_module._starimported_ignored_module_names)
                if target_module.code is None:
                    target_module_name = import_info[0]
                    source_module._starimported_ignored_module_names.add(target_module_name)
        source_module._deferred_imports = None

    def _find_module(self, name, path, parent=None):
        if False:
            print('Hello World!')
        '\n        3-tuple describing the physical location of the module with the passed\n        name if this module is physically findable _or_ raise `ImportError`.\n\n        This high-level method wraps the low-level `modulegraph.find_module()`\n        function with additional support for graph-based module caching.\n\n        Parameters\n        ----------\n        name : str\n            Fully-qualified name of the Python module to be found.\n        path : list\n            List of the absolute paths of all directories to search for this\n            module _or_ `None` if the default path list `self.path` is to be\n            searched.\n        parent : Node\n            Package containing this module if this module is a submodule of a\n            package _or_ `None` if this is a top-level module.\n\n        Returns\n        ----------\n        (filename, loader)\n            See `modulegraph._find_module()` for details.\n\n        Raises\n        ----------\n        ImportError\n            If this module is _not_ found.\n        '
        if parent is not None:
            fullname = parent.identifier + '.' + name
        else:
            fullname = name
        node = self.find_node(fullname)
        if node is not None:
            self.msg(3, 'find_module: already included?', node)
            raise ImportError(name)
        if path is None:
            if name in sys.builtin_module_names:
                return (None, BUILTIN_MODULE)
            path = self.path
        return self._find_module_path(fullname, name, path)

    def _find_module_path(self, fullname, module_name, search_dirs):
        if False:
            return 10
        '\n        3-tuple describing the physical location of the module with the passed\n        name if this module is physically findable _or_ raise `ImportError`.\n\n        This low-level function is a variant on the standard `imp.find_module()`\n        function with additional support for:\n\n        * Multiple search paths. The passed list of absolute paths will be\n          iteratively searched for the first directory containing a file\n          corresponding to this module.\n        * Compressed (e.g., zipped) packages.\n\n        For efficiency, the higher level `ModuleGraph._find_module()` method\n        wraps this function with support for module caching.\n\n        Parameters\n        ----------\n        module_name : str\n            Fully-qualified name of the module to be found.\n        search_dirs : list\n            List of the absolute paths of all directories to search for this\n            module (in order). Searching will halt at the first directory\n            containing this module.\n\n        Returns\n        ----------\n        (filename, loader)\n            2-tuple describing the physical location of this module, where:\n            * `filename` is the absolute path of this file.\n            * `loader` is the import loader.\n              In case of a namespace package, this is a NAMESPACE_PACKAGE\n              instance\n\n        Raises\n        ----------\n        ImportError\n            If this module is _not_ found.\n        '
        self.msgin(4, '_find_module_path <-', fullname, search_dirs)
        path_data = None
        namespace_dirs = []
        try:
            for search_dir in search_dirs:
                importer = pkgutil.get_importer(search_dir)
                if importer is None:
                    continue
                if hasattr(importer, 'find_spec'):
                    loader = None
                    spec = importer.find_spec(module_name)
                    if spec is not None:
                        loader = spec.loader
                        namespace_dirs.extend(spec.submodule_search_locations or [])
                elif hasattr(importer, 'find_loader'):
                    (loader, loader_namespace_dirs) = importer.find_loader(module_name)
                    namespace_dirs.extend(loader_namespace_dirs)
                elif hasattr(importer, 'find_module'):
                    loader = importer.find_module(module_name)
                else:
                    raise ImportError('Module %r importer %r loader unobtainable' % (module_name, importer))
                if loader is None:
                    continue
                pathname = None
                if hasattr(loader, 'get_filename'):
                    pathname = loader.get_filename(module_name)
                elif hasattr(loader, 'path'):
                    pathname = loader.path
                else:
                    raise ImportError('Module %r loader %r path unobtainable' % (module_name, loader))
                if pathname is None:
                    self.msg(4, '_find_module_path path not found', pathname)
                    continue
                path_data = (pathname, loader)
                break
            else:
                if namespace_dirs:
                    path_data = (namespace_dirs[0], NAMESPACE_PACKAGE(namespace_dirs))
        except UnicodeDecodeError as exc:
            self.msgout(1, '_find_module_path -> unicode error', exc)
        except Exception as exc:
            self.msgout(4, '_find_module_path -> exception', exc)
            raise
        self.msgout(4, '_find_module_path ->', path_data)
        if path_data is None:
            raise ImportError('No module named ' + repr(module_name))
        return path_data

    def create_xref(self, out=None):
        if False:
            return 10
        global header, footer, entry, contpl, contpl_linked, imports
        if out is None:
            out = sys.stdout
        scripts = []
        mods = []
        for mod in self.iter_graph():
            name = os.path.basename(mod.identifier)
            if isinstance(mod, Script):
                scripts.append((name, mod))
            else:
                mods.append((name, mod))
        scripts.sort()
        mods.sort()
        scriptnames = [sn for (sn, m) in scripts]
        scripts.extend(mods)
        mods = scripts
        title = 'modulegraph cross reference for ' + ', '.join(scriptnames)
        print(header % {'TITLE': title}, file=out)

        def sorted_namelist(mods):
            if False:
                print('Hello World!')
            lst = [os.path.basename(mod.identifier) for mod in mods if mod]
            lst.sort()
            return lst
        for (name, m) in mods:
            content = ''
            if isinstance(m, BuiltinModule):
                content = contpl % {'NAME': name, 'TYPE': '<i>(builtin module)</i>'}
            elif isinstance(m, Extension):
                content = contpl % {'NAME': name, 'TYPE': '<tt>%s</tt>' % m.filename}
            else:
                url = pathname2url(m.filename or '')
                content = contpl_linked % {'NAME': name, 'URL': url, 'TYPE': m.__class__.__name__}
            (oute, ince) = map(sorted_namelist, self.get_edges(m))
            if oute:
                links = []
                for n in oute:
                    links.append('  <a href="#%s">%s</a>\n' % (n, n))
                links = ' &#8226; '.join(links)
                content += imports % {'HEAD': 'imports', 'LINKS': links}
            if ince:
                links = []
                for n in ince:
                    links.append('  <a href="#%s">%s</a>\n' % (n, n))
                links = ' &#8226; '.join(links)
                content += imports % {'HEAD': 'imported by', 'LINKS': links}
            print(entry % {'NAME': name, 'CONTENT': content}, file=out)
        print(footer, file=out)

    def itergraphreport(self, name='G', flatpackages=()):
        if False:
            while True:
                i = 10
        nodes = list(map(self.graph.describe_node, self.graph.iterdfs(self)))
        describe_edge = self.graph.describe_edge
        edges = deque()
        packagenodes = set()
        packageidents = {}
        nodetoident = {}
        inpackages = {}
        mainedges = set()
        flatpackages = dict(flatpackages)

        def nodevisitor(node, data, outgoing, incoming):
            if False:
                print('Hello World!')
            if not isinstance(data, Node):
                return {'label': str(node)}
            s = '<f0> ' + type(data).__name__
            for (i, v) in enumerate(data.infoTuple()[:1], 1):
                s += '| <f%d> %s' % (i, v)
            return {'label': s, 'shape': 'record'}

        def edgevisitor(edge, data, head, tail):
            if False:
                i = 10
                return i + 15
            if data == 'orphan':
                return {'style': 'dashed'}
            elif data == 'pkgref':
                return {'style': 'dotted'}
            return {}
        yield ('digraph %s {\ncharset="UTF-8";\n' % (name,))
        attr = dict(rankdir='LR', concentrate='true')
        cpatt = '%s="%s"'
        for item in attr.items():
            yield ('\t%s;\n' % (cpatt % item,))
        for (node, data, outgoing, incoming) in nodes:
            nodetoident[node] = getattr(data, 'identifier', None)
            if isinstance(data, Package):
                packageidents[data.identifier] = node
                inpackages[node] = set([node])
                packagenodes.add(node)
        for (node, data, outgoing, incoming) in nodes:
            for edge in (describe_edge(e) for e in outgoing):
                edges.append(edge)
            yield ('\t"%s" [%s];\n' % (node, ','.join([cpatt % item for item in nodevisitor(node, data, outgoing, incoming).items()])))
            inside = inpackages.get(node)
            if inside is None:
                inside = inpackages[node] = set()
            ident = nodetoident[node]
            if ident is None:
                continue
            pkgnode = packageidents.get(ident[:ident.rfind('.')])
            if pkgnode is not None:
                inside.add(pkgnode)
        graph = []
        subgraphs = {}
        for key in packagenodes:
            subgraphs[key] = []
        while edges:
            (edge, data, head, tail) = edges.popleft()
            if (head, tail) in mainedges:
                continue
            mainedges.add((head, tail))
            tailpkgs = inpackages[tail]
            common = inpackages[head] & tailpkgs
            if not common and tailpkgs:
                usepkgs = sorted(tailpkgs)
                if len(usepkgs) != 1 or usepkgs[0] != tail:
                    edges.append((edge, data, head, usepkgs[0]))
                    edges.append((edge, 'pkgref', usepkgs[-1], tail))
                    continue
            if common:
                common = common.pop()
                if tail == common:
                    edges.append((edge, data, tail, head))
                elif head == common:
                    subgraphs[common].append((edge, 'pkgref', head, tail))
                else:
                    edges.append((edge, data, common, head))
                    edges.append((edge, data, common, tail))
            else:
                graph.append((edge, data, head, tail))

        def do_graph(edges, tabs):
            if False:
                while True:
                    i = 10
            edgestr = tabs + '"%s" -> "%s" [%s];\n'
            for (edge, data, head, tail) in edges:
                attribs = edgevisitor(edge, data, head, tail)
                yield (edgestr % (head, tail, ','.join([cpatt % item for item in attribs.items()])))
        for (g, edges) in subgraphs.items():
            yield ('\tsubgraph "cluster_%s" {\n' % (g,))
            yield ('\t\tlabel="%s";\n' % (nodetoident[g],))
            for s in do_graph(edges, '\t\t'):
                yield s
            yield '\t}\n'
        for s in do_graph(graph, '\t'):
            yield s
        yield '}\n'

    def graphreport(self, fileobj=None, flatpackages=()):
        if False:
            i = 10
            return i + 15
        if fileobj is None:
            fileobj = sys.stdout
        fileobj.writelines(self.itergraphreport(flatpackages=flatpackages))

    def report(self):
        if False:
            while True:
                i = 10
        'Print a report to stdout, listing the found modules with their\n        paths, as well as modules that are missing, or seem to be missing.\n        '
        print()
        print('%-15s %-25s %s' % ('Class', 'Name', 'File'))
        print('%-15s %-25s %s' % ('-----', '----', '----'))
        for m in sorted(self.iter_graph(), key=lambda n: n.identifier):
            print('%-15s %-25s %s' % (type(m).__name__, m.identifier, m.filename or ''))

    def _replace_paths_in_code(self, co):
        if False:
            for i in range(10):
                print('nop')
        new_filename = original_filename = os.path.normpath(co.co_filename)
        for (f, r) in self.replace_paths:
            f = os.path.join(f, '')
            r = os.path.join(r, '')
            if original_filename.startswith(f):
                new_filename = r + original_filename[len(f):]
                break
        else:
            return co
        consts = list(co.co_consts)
        for i in range(len(consts)):
            if isinstance(consts[i], type(co)):
                consts[i] = self._replace_paths_in_code(consts[i])
        code_func = type(co)
        return co.replace(co_consts=tuple(consts), co_filename=new_filename)