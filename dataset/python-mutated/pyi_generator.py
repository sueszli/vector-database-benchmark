"""The pyi generator module."""
import ast
import contextlib
import importlib
import inspect
import logging
import os
import re
import sys
import textwrap
from inspect import getfullargspec
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Type, get_args
import black
import black.mode
from reflex.components.component import Component
from reflex.utils import types as rx_types
from reflex.vars import Var
logger = logging.getLogger('pyi_generator')
EXCLUDED_FILES = ['__init__.py', 'component.py', 'bare.py', 'foreach.py', 'cond.py', 'multiselect.py', 'literals.py']
EXCLUDED_PROPS = ['alias', 'children', 'event_triggers', 'invalid_children', 'library', 'lib_dependencies', 'tag', 'is_default', 'special_props', 'valid_children']
DEFAULT_TYPING_IMPORTS = {'overload', 'Any', 'Dict', 'List', 'Literal', 'Optional', 'Union'}

def _get_type_hint(value, type_hint_globals, is_optional=True) -> str:
    if False:
        i = 10
        return i + 15
    'Resolve the type hint for value.\n\n    Args:\n        value: The type annotation as a str or actual types/aliases.\n        type_hint_globals: The globals to use to resolving a type hint str.\n        is_optional: Whether the type hint should be wrapped in Optional.\n\n    Returns:\n        The resolved type hint as a str.\n    '
    res = ''
    args = get_args(value)
    if args:
        inner_container_type_args = [repr(arg) for arg in args] if rx_types.is_literal(value) else [_get_type_hint(arg, type_hint_globals, is_optional=False) for arg in args if arg is not type(None)]
        res = f"{value.__name__}[{', '.join(inner_container_type_args)}]"
        if value.__name__ == 'Var':
            types = [res] + [_get_type_hint(arg, type_hint_globals, is_optional=False) for arg in args if arg is not type(None)]
            if len(types) > 1:
                res = ', '.join(types)
                res = f'Union[{res}]'
    elif isinstance(value, str):
        ev = eval(value, type_hint_globals)
        res = _get_type_hint(ev, type_hint_globals, is_optional=False) if ev.__name__ == 'Var' else value
    else:
        res = value.__name__
    if is_optional and (not res.startswith('Optional')):
        res = f'Optional[{res}]'
    return res

def _generate_imports(typing_imports: Iterable[str]) -> list[ast.ImportFrom]:
    if False:
        while True:
            i = 10
    'Generate the import statements for the stub file.\n\n    Args:\n        typing_imports: The typing imports to include.\n\n    Returns:\n        The list of import statements.\n    '
    return [ast.ImportFrom(module='typing', names=[ast.alias(name=imp) for imp in typing_imports]), *ast.parse(textwrap.dedent('\n                from reflex.vars import Var, BaseVar, ComputedVar\n                from reflex.event import EventChain, EventHandler, EventSpec\n                from reflex.style import Style')).body]

def _generate_docstrings(clzs: list[Type[Component]], props: list[str]) -> str:
    if False:
        print('Hello World!')
    'Generate the docstrings for the create method.\n\n    Args:\n        clzs: The classes to generate docstrings for.\n        props: The props to generate docstrings for.\n\n    Returns:\n        The docstring for the create method.\n    '
    props_comments = {}
    comments = []
    for clz in clzs:
        for line in inspect.getsource(clz).splitlines():
            reached_functions = re.search('def ', line)
            if reached_functions:
                break
            if line.strip().startswith('#'):
                comments.append(line)
                continue
            match = re.search('\\w+:', line)
            if match is None:
                continue
            prop = match.group(0).strip(':')
            if prop in props:
                if not comments:
                    continue
                props_comments[prop] = [comment.strip().strip('#') for comment in comments]
            comments.clear()
    clz = clzs[0]
    new_docstring = []
    for line in (clz.create.__doc__ or '').splitlines():
        if '**' in line:
            indent = line.split('**')[0]
            for nline in [f"{indent}{n}:{' '.join(c)}" for (n, c) in props_comments.items()]:
                new_docstring.append(nline)
        new_docstring.append(line)
    return '\n'.join(new_docstring)

def _extract_func_kwargs_as_ast_nodes(func: Callable, type_hint_globals: dict[str, Any]) -> list[tuple[ast.arg, ast.Constant | None]]:
    if False:
        for i in range(10):
            print('nop')
    'Get the kwargs already defined on the function.\n\n    Args:\n        func: The function to extract kwargs from.\n        type_hint_globals: The globals to use to resolving a type hint str.\n\n    Returns:\n        The list of kwargs as ast arg nodes.\n    '
    spec = getfullargspec(func)
    kwargs = []
    for kwarg in spec.kwonlyargs:
        arg = ast.arg(arg=kwarg)
        if kwarg in spec.annotations:
            arg.annotation = ast.Name(id=_get_type_hint(spec.annotations[kwarg], type_hint_globals))
        default = None
        if spec.kwonlydefaults is not None and kwarg in spec.kwonlydefaults:
            default = ast.Constant(value=spec.kwonlydefaults[kwarg])
        kwargs.append((arg, default))
    return kwargs

def _extract_class_props_as_ast_nodes(func: Callable, clzs: list[Type], type_hint_globals: dict[str, Any], extract_real_default: bool=False) -> list[tuple[ast.arg, ast.Constant | None]]:
    if False:
        print('Hello World!')
    'Get the props defined on the class and all parents.\n\n    Args:\n        func: The function that kwargs will be added to.\n        clzs: The classes to extract props from.\n        type_hint_globals: The globals to use to resolving a type hint str.\n        extract_real_default: Whether to extract the real default value from the\n            pydantic field definition.\n\n    Returns:\n        The list of props as ast arg nodes\n    '
    spec = getfullargspec(func)
    all_props = []
    kwargs = []
    for target_class in clzs:
        exec(f'from {target_class.__module__} import *', type_hint_globals)
        for (name, value) in target_class.__annotations__.items():
            if name in spec.kwonlyargs or name in EXCLUDED_PROPS or name in all_props:
                continue
            all_props.append(name)
            default = None
            if extract_real_default:
                with contextlib.suppress(AttributeError, KeyError):
                    default = target_class.__fields__[name].default
                    if isinstance(default, Var):
                        default = default._decode()
            kwargs.append((ast.arg(arg=name, annotation=ast.Name(id=_get_type_hint(value, type_hint_globals))), ast.Constant(value=default)))
    return kwargs

def _generate_component_create_functiondef(node: ast.FunctionDef | None, clz: type[Component], type_hint_globals: dict[str, Any]) -> ast.FunctionDef:
    if False:
        print('Hello World!')
    'Generate the create function definition for a Component.\n\n    Args:\n        node: The existing create functiondef node from the ast\n        clz: The Component class to generate the create functiondef for.\n        type_hint_globals: The globals to use to resolving a type hint str.\n\n    Returns:\n        The create functiondef node for the ast.\n    '
    kwargs = _extract_func_kwargs_as_ast_nodes(clz.create, type_hint_globals)
    all_classes = [c for c in clz.__mro__ if issubclass(c, Component)]
    prop_kwargs = _extract_class_props_as_ast_nodes(clz.create, all_classes, type_hint_globals)
    all_props = [arg[0].arg for arg in prop_kwargs]
    kwargs.extend(prop_kwargs)
    kwargs.extend(((ast.arg(arg=trigger, annotation=ast.Name(id='Optional[Union[EventHandler, EventSpec, List, function, BaseVar]]')), ast.Constant(value=None)) for trigger in sorted(clz().get_event_triggers().keys())))
    logger.debug(f'Generated {clz.__name__}.create method with {len(kwargs)} kwargs')
    create_args = ast.arguments(args=[ast.arg(arg='cls')], posonlyargs=[], vararg=ast.arg(arg='children'), kwonlyargs=[arg[0] for arg in kwargs], kw_defaults=[arg[1] for arg in kwargs], kwarg=ast.arg(arg='props'), defaults=[])
    definition = ast.FunctionDef(name='create', args=create_args, body=[ast.Expr(value=ast.Constant(value=_generate_docstrings(all_classes, all_props))), ast.Expr(value=ast.Ellipsis())], decorator_list=[ast.Name(id='overload'), *(node.decorator_list if node is not None else [ast.Name(id='classmethod')])], lineno=node.lineno if node is not None else None, returns=ast.Constant(value=clz.__name__))
    return definition

class StubGenerator(ast.NodeTransformer):
    """A node transformer that will generate the stubs for a given module."""

    def __init__(self, module: ModuleType, classes: dict[str, Type[Component]]):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the stub generator.\n\n        Args:\n            module: The actual module object module to generate stubs for.\n            classes: The actual Component class objects to generate stubs for.\n        '
        super().__init__()
        self.classes = classes
        self.current_class = None
        self.typing_imports = DEFAULT_TYPING_IMPORTS
        self.inserted_imports = False
        self.import_statements: list[str] = []
        self.type_hint_globals = module.__dict__.copy()

    @staticmethod
    def _remove_docstring(node: ast.Module | ast.ClassDef | ast.FunctionDef) -> ast.Module | ast.ClassDef | ast.FunctionDef:
        if False:
            for i in range(10):
                print('nop')
        'Removes any docstring in place.\n\n        Args:\n            node: The node to remove the docstring from.\n\n        Returns:\n            The modified node.\n        '
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            node.body.pop(0)
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        if False:
            for i in range(10):
                print('nop')
        'Visit a Module node and remove docstring from body.\n\n        Args:\n            node: The Module node to visit.\n\n        Returns:\n            The modified Module node.\n        '
        self.generic_visit(node)
        return self._remove_docstring(node)

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> ast.Import | ast.ImportFrom | list[ast.Import | ast.ImportFrom]:
        if False:
            while True:
                i = 10
        'Collect import statements from the module.\n\n        If this is the first import statement, insert the typing imports before it.\n\n        Args:\n            node: The import node to visit.\n\n        Returns:\n            The modified import node(s).\n        '
        self.import_statements.append(ast.unparse(node))
        if not self.inserted_imports:
            self.inserted_imports = True
            return _generate_imports(self.typing_imports) + [node]
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.Import | ast.ImportFrom | list[ast.Import | ast.ImportFrom] | None:
        if False:
            while True:
                i = 10
        'Visit an ImportFrom node.\n\n        Remove any `from __future__ import *` statements, and hand off to visit_Import.\n\n        Args:\n            node: The ImportFrom node to visit.\n\n        Returns:\n            The modified ImportFrom node.\n        '
        if node.module == '__future__':
            return None
        return self.visit_Import(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        if False:
            print('Hello World!')
        'Visit a ClassDef node.\n\n        Remove all assignments in the class body, and add a create functiondef\n        if one does not exist.\n\n        Args:\n            node: The ClassDef node to visit.\n\n        Returns:\n            The modified ClassDef node.\n        '
        exec('\n'.join(self.import_statements), self.type_hint_globals)
        self.current_class = node.name
        self._remove_docstring(node)
        self.generic_visit(node)
        if not node.body:
            node.body.append(ast.Expr(value=ast.Ellipsis()))
        if not any((isinstance(child, ast.FunctionDef) and child.name == 'create' for child in node.body)) and self.current_class in self.classes:
            node.body.append(_generate_component_create_functiondef(node=None, clz=self.classes[self.current_class], type_hint_globals=self.type_hint_globals))
        self.current_class = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if False:
            return 10
        'Visit a FunctionDef node.\n\n        Special handling for `.create` functions to add type hints for all props\n        defined on the component class.\n\n        Remove all private functions and blank out the function body of the\n        remaining public functions.\n\n        Args:\n            node: The FunctionDef node to visit.\n\n        Returns:\n            The modified FunctionDef node (or None).\n        '
        if node.name == 'create' and self.current_class in self.classes:
            node = _generate_component_create_functiondef(node, self.classes[self.current_class], self.type_hint_globals)
        else:
            if node.name.startswith('_'):
                return None
            node.body = [ast.Expr(value=ast.Ellipsis())]
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
        if False:
            for i in range(10):
                print('nop')
        'Remove non-annotated assignment statements.\n\n        Args:\n            node: The Assign node to visit.\n\n        Returns:\n            The modified Assign node (or None).\n        '
        if node.value is not None and isinstance(node.value, ast.Name) and (node.value.id == 'Any'):
            return node
        return None

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign | None:
        if False:
            print('Hello World!')
        'Visit an AnnAssign node (Annotated assignment).\n\n        Remove private target and remove the assignment value in the stub.\n\n        Args:\n            node: The AnnAssign node to visit.\n\n        Returns:\n            The modified AnnAssign node (or None).\n        '
        if isinstance(node.target, ast.Name) and node.target.id.startswith('_'):
            return None
        if self.current_class in self.classes:
            return None
        node.value = None
        return node

class PyiGenerator:
    """A .pyi file generator that will scan all defined Component in Reflex and
    generate the approriate stub.
    """
    modules: list = []
    root: str = ''
    current_module: Any = {}
    default_typing_imports: set = DEFAULT_TYPING_IMPORTS

    def _write_pyi_file(self, module_path: Path, source: str):
        if False:
            return 10
        pyi_content = [f'"""Stub file for {module_path}"""', '# ------------------- DO NOT EDIT ----------------------', '# This file was generated by `scripts/pyi_generator.py`!', '# ------------------------------------------------------', '']
        for formatted_line in black.format_file_contents(src_contents=source, fast=True, mode=black.mode.Mode(is_pyi=True)).splitlines():
            if formatted_line == '    def create(':
                pyi_content.append('    def create(  # type: ignore')
            else:
                pyi_content.append(formatted_line)
        pyi_path = module_path.with_suffix('.pyi')
        pyi_path.write_text('\n'.join(pyi_content))
        logger.info(f'Wrote {pyi_path}')

    def _scan_file(self, module_path: Path):
        if False:
            return 10
        module_import = str(module_path.with_suffix('')).replace('/', '.')
        module = importlib.import_module(module_import)
        class_names = {name: obj for (name, obj) in vars(module).items() if inspect.isclass(obj) and issubclass(obj, Component) and (obj != Component) and (inspect.getmodule(obj) == module)}
        if not class_names:
            return
        new_tree = StubGenerator(module, class_names).visit(ast.parse(inspect.getsource(module)))
        self._write_pyi_file(module_path, ast.unparse(new_tree))

    def _scan_folder(self, folder):
        if False:
            i = 10
            return i + 15
        for (root, _, files) in os.walk(folder):
            for file in files:
                if file in EXCLUDED_FILES:
                    continue
                if file.endswith('.py'):
                    self._scan_file(Path(root) / file)

    def scan_all(self, targets):
        if False:
            return 10
        'Scan all targets for class inheriting Component and generate the .pyi files.\n\n        Args:\n            targets: the list of file/folders to scan.\n        '
        for target in targets:
            if target.endswith('.py'):
                self._scan_file(Path(target))
            else:
                self._scan_folder(target)

def generate_init():
    if False:
        return 10
    'Generate a pyi file for the main __init__.py.'
    from reflex import _MAPPING
    imports = [f"from {(path if mod != path.rsplit('.')[-1] or mod == 'page' else '.'.join(path.rsplit('.')[:-1]))} import {mod} as {mod}" for (mod, path) in _MAPPING.items()]
    with open('reflex/__init__.pyi', 'w') as pyi_file:
        pyi_file.writelines('\n'.join(imports))
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('blib2to3.pgen2.driver').setLevel(logging.INFO)
    targets = sys.argv[1:] if len(sys.argv) > 1 else ['reflex/components']
    logger.info(f'Running .pyi generator for {targets}')
    gen = PyiGenerator()
    gen.scan_all(targets)
    generate_init()