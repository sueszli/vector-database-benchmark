"""Utilities for mypy.stubgen, mypy.stubgenc, and mypy.stubdoc modules."""
from __future__ import annotations
import os.path
import re
import sys
from abc import abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from typing import Final, Iterable, Iterator, Mapping
from typing_extensions import overload
from mypy_extensions import mypyc_attr
import mypy.options
from mypy.modulefinder import ModuleNotFoundReason
from mypy.moduleinspect import InspectError, ModuleInspect
from mypy.stubdoc import ArgSig, FunctionSig
from mypy.types import AnyType, NoneType, Type, TypeList, TypeStrVisitor, UnboundType, UnionType
NOT_IMPORTABLE_MODULES = ()

class CantImport(Exception):

    def __init__(self, module: str, message: str) -> None:
        if False:
            i = 10
            return i + 15
        self.module = module
        self.message = message

def walk_packages(inspect: ModuleInspect, packages: list[str], verbose: bool=False) -> Iterator[str]:
    if False:
        i = 10
        return i + 15
    'Iterates through all packages and sub-packages in the given list.\n\n    This uses runtime imports (in another process) to find both Python and C modules.\n    For Python packages we simply pass the __path__ attribute to pkgutil.walk_packages() to\n    get the content of the package (all subpackages and modules).  However, packages in C\n    extensions do not have this attribute, so we have to roll out our own logic: recursively\n    find all modules imported in the package that have matching names.\n    '
    for package_name in packages:
        if package_name in NOT_IMPORTABLE_MODULES:
            print(f'{package_name}: Skipped (blacklisted)')
            continue
        if verbose:
            print(f'Trying to import {package_name!r} for runtime introspection')
        try:
            prop = inspect.get_package_properties(package_name)
        except InspectError:
            report_missing(package_name)
            continue
        yield prop.name
        if prop.is_c_module:
            yield from walk_packages(inspect, prop.subpackages, verbose)
        else:
            yield from prop.subpackages

def find_module_path_using_sys_path(module: str, sys_path: list[str]) -> str | None:
    if False:
        while True:
            i = 10
    relative_candidates = (module.replace('.', '/') + '.py', os.path.join(module.replace('.', '/'), '__init__.py'))
    for base in sys_path:
        for relative_path in relative_candidates:
            path = os.path.join(base, relative_path)
            if os.path.isfile(path):
                return path
    return None

def find_module_path_and_all_py3(inspect: ModuleInspect, module: str, verbose: bool) -> tuple[str | None, list[str] | None] | None:
    if False:
        for i in range(10):
            print('nop')
    'Find module and determine __all__ for a Python 3 module.\n\n    Return None if the module is a C or pyc-only module.\n    Return (module_path, __all__) if it is a Python module.\n    Raise CantImport if import failed.\n    '
    if module in NOT_IMPORTABLE_MODULES:
        raise CantImport(module, '')
    if verbose:
        print(f'Trying to import {module!r} for runtime introspection')
    try:
        mod = inspect.get_package_properties(module)
    except InspectError as e:
        path = find_module_path_using_sys_path(module, sys.path)
        if path is None:
            raise CantImport(module, str(e)) from e
        return (path, None)
    if mod.is_c_module:
        return None
    return (mod.file, mod.all)

@contextmanager
def generate_guarded(mod: str, target: str, ignore_errors: bool=True, verbose: bool=False) -> Iterator[None]:
    if False:
        print('Hello World!')
    'Ignore or report errors during stub generation.\n\n    Optionally report success.\n    '
    if verbose:
        print(f'Processing {mod}')
    try:
        yield
    except Exception as e:
        if not ignore_errors:
            raise e
        else:
            print('Stub generation failed for', mod, file=sys.stderr)
    else:
        if verbose:
            print(f'Created {target}')

def report_missing(mod: str, message: str | None='', traceback: str='') -> None:
    if False:
        print('Hello World!')
    if message:
        message = ' with error: ' + message
    print(f'{mod}: Failed to import, skipping{message}')

def fail_missing(mod: str, reason: ModuleNotFoundReason) -> None:
    if False:
        while True:
            i = 10
    if reason is ModuleNotFoundReason.NOT_FOUND:
        clarification = '(consider using --search-path)'
    elif reason is ModuleNotFoundReason.FOUND_WITHOUT_TYPE_HINTS:
        clarification = '(module likely exists, but is not PEP 561 compatible)'
    else:
        clarification = f"(unknown reason '{reason}')"
    raise SystemExit(f"Can't find module '{mod}' {clarification}")

@overload
def remove_misplaced_type_comments(source: bytes) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def remove_misplaced_type_comments(source: str) -> str:
    if False:
        print('Hello World!')
    ...

def remove_misplaced_type_comments(source: str | bytes) -> str | bytes:
    if False:
        return 10
    'Remove comments from source that could be understood as misplaced type comments.\n\n    Normal comments may look like misplaced type comments, and since they cause blocking\n    parse errors, we want to avoid them.\n    '
    if isinstance(source, bytes):
        text = source.decode('latin1')
    else:
        text = source
    text = re.sub('^[ \\t]*# +type: +["\\\'a-zA-Z_].*$', '', text, flags=re.MULTILINE)
    text = re.sub('""" *\\n[ \\t\\n]*# +type: +\\(.*$', '"""\n', text, flags=re.MULTILINE)
    text = re.sub("''' *\\n[ \\t\\n]*# +type: +\\(.*$", "'''\n", text, flags=re.MULTILINE)
    text = re.sub('^[ \\t]*# +type: +\\([^()]+(\\)[ \\t]*)?$', '', text, flags=re.MULTILINE)
    if isinstance(source, bytes):
        return text.encode('latin1')
    else:
        return text

def common_dir_prefix(paths: list[str]) -> str:
    if False:
        return 10
    if not paths:
        return '.'
    cur = os.path.dirname(os.path.normpath(paths[0]))
    for path in paths[1:]:
        while True:
            path = os.path.dirname(os.path.normpath(path))
            if (cur + os.sep).startswith(path + os.sep):
                cur = path
                break
    return cur or '.'

class AnnotationPrinter(TypeStrVisitor):
    """Visitor used to print existing annotations in a file.

    The main difference from TypeStrVisitor is a better treatment of
    unbound types.

    Notes:
    * This visitor doesn't add imports necessary for annotations, this is done separately
      by ImportTracker.
    * It can print all kinds of types, but the generated strings may not be valid (notably
      callable types) since it prints the same string that reveal_type() does.
    * For Instance types it prints the fully qualified names.
    """

    def __init__(self, stubgen: BaseStubGenerator, known_modules: list[str] | None=None, local_modules: list[str] | None=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(options=mypy.options.Options())
        self.stubgen = stubgen
        self.known_modules = known_modules
        self.local_modules = local_modules or ['builtins']

    def visit_any(self, t: AnyType) -> str:
        if False:
            i = 10
            return i + 15
        s = super().visit_any(t)
        self.stubgen.import_tracker.require_name(s)
        return s

    def visit_unbound_type(self, t: UnboundType) -> str:
        if False:
            i = 10
            return i + 15
        s = t.name
        if self.known_modules is not None and '.' in s:
            for module_name in self.local_modules + sorted(self.known_modules, reverse=True):
                if s.startswith(module_name + '.'):
                    if module_name in self.local_modules:
                        s = s[len(module_name) + 1:]
                    arg_module = module_name
                    break
            else:
                arg_module = s[:s.rindex('.')]
            if arg_module not in self.local_modules:
                self.stubgen.import_tracker.add_import(arg_module, require=True)
        elif s == 'NoneType':
            s = 'None'
        else:
            self.stubgen.import_tracker.require_name(s)
        if t.args:
            s += f'[{self.args_str(t.args)}]'
        return s

    def visit_none_type(self, t: NoneType) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'None'

    def visit_type_list(self, t: TypeList) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'[{self.list_str(t.items)}]'

    def visit_union_type(self, t: UnionType) -> str:
        if False:
            i = 10
            return i + 15
        return ' | '.join([item.accept(self) for item in t.items])

    def args_str(self, args: Iterable[Type]) -> str:
        if False:
            return 10
        'Convert an array of arguments to strings and join the results with commas.\n\n        The main difference from list_str is the preservation of quotes for string\n        arguments\n        '
        types = ['builtins.bytes', 'builtins.str']
        res = []
        for arg in args:
            arg_str = arg.accept(self)
            if isinstance(arg, UnboundType) and arg.original_str_fallback in types:
                res.append(f"'{arg_str}'")
            else:
                res.append(arg_str)
        return ', '.join(res)

class ClassInfo:

    def __init__(self, name: str, self_var: str, docstring: str | None=None, cls: type | None=None) -> None:
        if False:
            print('Hello World!')
        self.name = name
        self.self_var = self_var
        self.docstring = docstring
        self.cls = cls

class FunctionContext:

    def __init__(self, module_name: str, name: str, docstring: str | None=None, is_abstract: bool=False, class_info: ClassInfo | None=None) -> None:
        if False:
            return 10
        self.module_name = module_name
        self.name = name
        self.docstring = docstring
        self.is_abstract = is_abstract
        self.class_info = class_info
        self._fullname: str | None = None

    @property
    def fullname(self) -> str:
        if False:
            return 10
        if self._fullname is None:
            if self.class_info:
                self._fullname = f'{self.module_name}.{self.class_info.name}.{self.name}'
            else:
                self._fullname = f'{self.module_name}.{self.name}'
        return self._fullname

def infer_method_ret_type(name: str) -> str | None:
    if False:
        return 10
    'Infer return types for known special methods'
    if name.startswith('__') and name.endswith('__'):
        name = name[2:-2]
        if name in ('float', 'bool', 'bytes', 'int', 'complex', 'str'):
            return name
        elif name in ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'contains'):
            return 'bool'
        elif name in ('len', 'length_hint', 'index', 'hash', 'sizeof', 'trunc', 'floor', 'ceil'):
            return 'int'
        elif name in ('format', 'repr'):
            return 'str'
        elif name in ('init', 'setitem', 'del', 'delitem'):
            return 'None'
    return None

def infer_method_arg_types(name: str, self_var: str='self', arg_names: list[str] | None=None) -> list[ArgSig] | None:
    if False:
        while True:
            i = 10
    'Infer argument types for known special methods'
    args: list[ArgSig] | None = None
    if name.startswith('__') and name.endswith('__'):
        if arg_names and len(arg_names) >= 1 and (arg_names[0] == 'self'):
            arg_names = arg_names[1:]
        name = name[2:-2]
        if name == 'exit':
            if arg_names is None:
                arg_names = ['type', 'value', 'traceback']
            if len(arg_names) == 3:
                arg_types = ['type[BaseException] | None', 'BaseException | None', 'types.TracebackType | None']
                args = [ArgSig(name=arg_name, type=arg_type) for (arg_name, arg_type) in zip(arg_names, arg_types)]
    if args is not None:
        return [ArgSig(name=self_var)] + args
    return None

@mypyc_attr(allow_interpreted_subclasses=True)
class SignatureGenerator:
    """Abstract base class for extracting a list of FunctionSigs for each function."""

    def remove_self_type(self, inferred: list[FunctionSig] | None, self_var: str) -> list[FunctionSig] | None:
        if False:
            for i in range(10):
                print('nop')
        'Remove type annotation from self/cls argument'
        if inferred:
            for signature in inferred:
                if signature.args:
                    if signature.args[0].name == self_var:
                        signature.args[0].type = None
        return inferred

    @abstractmethod
    def get_function_sig(self, default_sig: FunctionSig, ctx: FunctionContext) -> list[FunctionSig] | None:
        if False:
            return 10
        'Return a list of signatures for the given function.\n\n        If no signature can be found, return None. If all of the registered SignatureGenerators\n        for the stub generator return None, then the default_sig will be used.\n        '
        pass

    @abstractmethod
    def get_property_type(self, default_type: str | None, ctx: FunctionContext) -> str | None:
        if False:
            i = 10
            return i + 15
        'Return the type of the given property'
        pass

class ImportTracker:
    """Record necessary imports during stub generation."""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.module_for: dict[str, str | None] = {}
        self.direct_imports: dict[str, str] = {}
        self.reverse_alias: dict[str, str] = {}
        self.required_names: set[str] = set()
        self.reexports: set[str] = set()

    def add_import_from(self, module: str, names: list[tuple[str, str | None]], require: bool=False) -> None:
        if False:
            print('Hello World!')
        for (name, alias) in names:
            if alias:
                self.module_for[alias] = module
                self.reverse_alias[alias] = name
            else:
                self.module_for[name] = module
                self.reverse_alias.pop(name, None)
            if require:
                self.require_name(alias or name)
            self.direct_imports.pop(alias or name, None)

    def add_import(self, module: str, alias: str | None=None, require: bool=False) -> None:
        if False:
            return 10
        if alias:
            assert '.' not in alias
            self.module_for[alias] = None
            self.reverse_alias[alias] = module
            if require:
                self.required_names.add(alias)
        else:
            name = module
            if require:
                self.required_names.add(name)
            while name:
                self.module_for[name] = None
                self.direct_imports[name] = module
                self.reverse_alias.pop(name, None)
                name = name.rpartition('.')[0]

    def require_name(self, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        while name not in self.direct_imports and '.' in name:
            name = name.rsplit('.', 1)[0]
        self.required_names.add(name)

    def reexport(self, name: str) -> None:
        if False:
            print('Hello World!')
        'Mark a given non qualified name as needed in __all__.\n\n        This means that in case it comes from a module, it should be\n        imported with an alias even if the alias is the same as the name.\n        '
        self.require_name(name)
        self.reexports.add(name)

    def import_lines(self) -> list[str]:
        if False:
            return 10
        "The list of required import lines (as strings with python code).\n\n        In order for a module be included in this output, an indentifier must be both\n        'required' via require_name() and 'imported' via add_import_from()\n        or add_import()\n        "
        result = []
        module_map: Mapping[str, list[str]] = defaultdict(list)
        for name in sorted(self.required_names, key=lambda n: (self.reverse_alias[n], n) if n in self.reverse_alias else (n, '')):
            if name not in self.module_for:
                continue
            m = self.module_for[name]
            if m is not None:
                if name in self.reverse_alias:
                    name = f'{self.reverse_alias[name]} as {name}'
                elif name in self.reexports:
                    name = f'{name} as {name}'
                module_map[m].append(name)
            elif name in self.reverse_alias:
                source = self.reverse_alias[name]
                result.append(f'import {source} as {name}\n')
            elif name in self.reexports:
                assert '.' not in name
                result.append(f'import {name} as {name}\n')
            else:
                result.append(f'import {name}\n')
        for (module, names) in sorted(module_map.items()):
            result.append(f"from {module} import {', '.join(sorted(names))}\n")
        return result

@mypyc_attr(allow_interpreted_subclasses=True)
class BaseStubGenerator:
    IGNORED_DUNDERS: Final = {'__all__', '__author__', '__about__', '__copyright__', '__email__', '__license__', '__summary__', '__title__', '__uri__', '__str__', '__repr__', '__getstate__', '__setstate__', '__slots__', '__builtins__', '__cached__', '__file__', '__name__', '__package__', '__path__', '__spec__', '__loader__'}
    TYPING_MODULE_NAMES: Final = ('typing', 'typing_extensions')
    EXTRA_EXPORTED: Final = {'pyasn1_modules.rfc2437.univ', 'pyasn1_modules.rfc2459.char', 'pyasn1_modules.rfc2459.univ'}

    def __init__(self, _all_: list[str] | None=None, include_private: bool=False, export_less: bool=False, include_docstrings: bool=False):
        if False:
            i = 10
            return i + 15
        self._all_ = _all_
        self._include_private = include_private
        self._include_docstrings = include_docstrings
        self.export_less = export_less
        self._import_lines: list[str] = []
        self._output: list[str] = []
        self._indent = ''
        self._toplevel_names: list[str] = []
        self.import_tracker = ImportTracker()
        self.defined_names: set[str] = set()
        self.sig_generators = self.get_sig_generators()
        self.module_name: str = ''

    def get_sig_generators(self) -> list[SignatureGenerator]:
        if False:
            for i in range(10):
                print('nop')
        return []

    def refers_to_fullname(self, name: str, fullname: str | tuple[str, ...]) -> bool:
        if False:
            print('Hello World!')
        'Return True if the variable name identifies the same object as the given fullname(s).'
        if isinstance(fullname, tuple):
            return any((self.refers_to_fullname(name, fname) for fname in fullname))
        (module, short) = fullname.rsplit('.', 1)
        return self.import_tracker.module_for.get(name) == module and (name == short or self.import_tracker.reverse_alias.get(name) == short)

    def add_name(self, fullname: str, require: bool=True) -> str:
        if False:
            print('Hello World!')
        "Add a name to be imported and return the name reference.\n\n        The import will be internal to the stub (i.e don't reexport).\n        "
        (module, name) = fullname.rsplit('.', 1)
        alias = '_' + name if name in self.defined_names else None
        self.import_tracker.add_import_from(module, [(name, alias)], require=require)
        return alias or name

    def add_import_line(self, line: str) -> None:
        if False:
            i = 10
            return i + 15
        "Add a line of text to the import section, unless it's already there."
        if line not in self._import_lines:
            self._import_lines.append(line)

    def get_imports(self) -> str:
        if False:
            while True:
                i = 10
        'Return the import statements for the stub.'
        imports = ''
        if self._import_lines:
            imports += ''.join(self._import_lines)
        imports += ''.join(self.import_tracker.import_lines())
        return imports

    def output(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return the text for the stub.'
        pieces: list[str] = []
        if (imports := self.get_imports()):
            pieces.append(imports)
        if (dunder_all := self.get_dunder_all()):
            pieces.append(dunder_all)
        if self._output:
            pieces.append(''.join(self._output))
        return '\n'.join(pieces)

    def get_dunder_all(self) -> str:
        if False:
            return 10
        'Return the __all__ list for the stub.'
        if self._all_:
            return f'__all__ = {self._all_!r}\n'
        return ''

    def add(self, string: str) -> None:
        if False:
            return 10
        'Add text to generated stub.'
        self._output.append(string)

    def is_top_level(self) -> bool:
        if False:
            return 10
        'Are we processing the top level of a file?'
        return self._indent == ''

    def indent(self) -> None:
        if False:
            return 10
        'Add one level of indentation.'
        self._indent += '    '

    def dedent(self) -> None:
        if False:
            return 10
        'Remove one level of indentation.'
        self._indent = self._indent[:-4]

    def record_name(self, name: str) -> None:
        if False:
            i = 10
            return i + 15
        'Mark a name as defined.\n\n        This only does anything if at the top level of a module.\n        '
        if self.is_top_level():
            self._toplevel_names.append(name)

    def is_recorded_name(self, name: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Has this name been recorded previously?'
        return self.is_top_level() and name in self._toplevel_names

    def set_defined_names(self, defined_names: set[str]) -> None:
        if False:
            while True:
                i = 10
        self.defined_names = defined_names
        for name in self._all_ or ():
            self.import_tracker.reexport(name)
        known_imports = {'_typeshed': ['Incomplete'], 'typing': ['Any', 'TypeVar', 'NamedTuple'], 'collections.abc': ['Generator'], 'typing_extensions': ['TypedDict', 'ParamSpec', 'TypeVarTuple']}
        for (pkg, imports) in known_imports.items():
            for t in imports:
                self.add_name(f'{pkg}.{t}', require=False)

    def check_undefined_names(self) -> None:
        if False:
            print('Hello World!')
        undefined_names = [name for name in self._all_ or [] if name not in self._toplevel_names]
        if undefined_names:
            if self._output:
                self.add('\n')
            self.add('# Names in __all__ with no definition:\n')
            for name in sorted(undefined_names):
                self.add(f'#   {name}\n')

    def get_signatures(self, default_signature: FunctionSig, sig_generators: list[SignatureGenerator], func_ctx: FunctionContext) -> list[FunctionSig]:
        if False:
            print('Hello World!')
        for sig_gen in sig_generators:
            inferred = sig_gen.get_function_sig(default_signature, func_ctx)
            if inferred:
                return inferred
        return [default_signature]

    def get_property_type(self, default_type: str | None, sig_generators: list[SignatureGenerator], func_ctx: FunctionContext) -> str | None:
        if False:
            print('Hello World!')
        for sig_gen in sig_generators:
            inferred = sig_gen.get_property_type(default_type, func_ctx)
            if inferred:
                return inferred
        return default_type

    def format_func_def(self, sigs: list[FunctionSig], is_coroutine: bool=False, decorators: list[str] | None=None, docstring: str | None=None) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        lines: list[str] = []
        if decorators is None:
            decorators = []
        for signature in sigs:
            for deco in decorators:
                lines.append(f'{self._indent}{deco}')
            lines.append(signature.format_sig(indent=self._indent, is_async=is_coroutine, docstring=docstring if self._include_docstrings else None))
        return lines

    def print_annotation(self, t: Type, known_modules: list[str] | None=None, local_modules: list[str] | None=None) -> str:
        if False:
            i = 10
            return i + 15
        printer = AnnotationPrinter(self, known_modules, local_modules)
        return t.accept(printer)

    def is_not_in_all(self, name: str) -> bool:
        if False:
            return 10
        if self.is_private_name(name):
            return False
        if self._all_:
            return self.is_top_level() and name not in self._all_
        return False

    def is_private_name(self, name: str, fullname: str | None=None) -> bool:
        if False:
            i = 10
            return i + 15
        if self._include_private:
            return False
        if fullname in self.EXTRA_EXPORTED:
            return False
        if name == '_':
            return False
        if not name.startswith('_'):
            return False
        if self._all_ and name in self._all_:
            return False
        if name.startswith('__') and name.endswith('__'):
            return name in self.IGNORED_DUNDERS
        return True

    def should_reexport(self, name: str, full_module: str, name_is_alias: bool) -> bool:
        if False:
            print('Hello World!')
        if not name_is_alias and self.module_name and (self.module_name + '.' + name in self.EXTRA_EXPORTED):
            return True
        if name_is_alias:
            return False
        if self.export_less:
            return False
        if not self.module_name:
            return False
        is_private = self.is_private_name(name, full_module + '.' + name)
        if is_private:
            return False
        top_level = full_module.split('.')[0]
        self_top_level = self.module_name.split('.', 1)[0]
        if top_level not in (self_top_level, '_' + self_top_level):
            return False
        if self._all_:
            return name in self._all_
        return True