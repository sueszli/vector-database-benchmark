"""Preserve function defaults.

Preserve the default argument values of function signatures in source code
and keep them not evaluated for readability.
"""
from __future__ import annotations
import ast
import inspect
import types
import warnings
from typing import TYPE_CHECKING
import sphinx
from sphinx.deprecation import RemovedInSphinx90Warning
from sphinx.locale import __
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
if TYPE_CHECKING:
    from typing import Any
    from sphinx.application import Sphinx
logger = logging.getLogger(__name__)
_LAMBDA_NAME = (lambda : None).__name__

class DefaultValue:

    def __init__(self, name: str) -> None:
        if False:
            return 10
        self.name = name

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.name

def get_function_def(obj: Any) -> ast.FunctionDef | None:
    if False:
        for i in range(10):
            print('nop')
    'Get FunctionDef object from living object.\n\n    This tries to parse original code for living object and returns\n    AST node for given *obj*.\n    '
    warnings.warn('sphinx.ext.autodoc.preserve_defaults.get_function_def is deprecated and scheduled for removal in Sphinx 9. Use sphinx.ext.autodoc.preserve_defaults._get_arguments() to extract AST arguments objects from a lambda or regular function.', RemovedInSphinx90Warning, stacklevel=2)
    try:
        source = inspect.getsource(obj)
        if source.startswith((' ', '\t')):
            module = ast.parse('if True:\n' + source)
            return module.body[0].body[0]
        else:
            module = ast.parse(source)
            return module.body[0]
    except (OSError, TypeError):
        return None

def _get_arguments(obj: Any, /) -> ast.arguments | None:
    if False:
        for i in range(10):
            print('nop')
    "Parse 'ast.arguments' from an object.\n\n    This tries to parse the original code for an object and returns\n    an 'ast.arguments' node.\n    "
    try:
        source = inspect.getsource(obj)
        if source.startswith((' ', '\t')):
            module = ast.parse('if True:\n' + source)
            subject = module.body[0].body[0]
        else:
            module = ast.parse(source)
            subject = module.body[0]
    except (OSError, TypeError):
        return None
    except SyntaxError:
        if _is_lambda(obj):
            return None
        raise
    return _get_arguments_inner(subject)

def _is_lambda(x, /):
    if False:
        print('Hello World!')
    return isinstance(x, types.LambdaType) and x.__name__ == _LAMBDA_NAME

def _get_arguments_inner(x: Any, /) -> ast.arguments | None:
    if False:
        i = 10
        return i + 15
    if isinstance(x, (ast.AsyncFunctionDef, ast.FunctionDef, ast.Lambda)):
        return x.args
    if isinstance(x, (ast.Assign, ast.AnnAssign)):
        return _get_arguments_inner(x.value)
    return None

def get_default_value(lines: list[str], position: ast.AST) -> str | None:
    if False:
        for i in range(10):
            print('nop')
    try:
        if position.lineno == position.end_lineno:
            line = lines[position.lineno - 1]
            return line[position.col_offset:position.end_col_offset]
        else:
            return None
    except (AttributeError, IndexError):
        return None

def update_defvalue(app: Sphinx, obj: Any, bound_method: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Update defvalue info of *obj* using type_comments.'
    if not app.config.autodoc_preserve_defaults:
        return
    try:
        lines = inspect.getsource(obj).splitlines()
        if lines[0].startswith((' ', '\t')):
            lines.insert(0, '')
    except (OSError, TypeError):
        lines = []
    try:
        args = _get_arguments(obj)
    except SyntaxError:
        return
    if args is None:
        return
    if not args.defaults and (not args.kw_defaults):
        return
    try:
        if bound_method and inspect.ismethod(obj) and hasattr(obj, '__func__'):
            sig = inspect.signature(obj.__func__)
        else:
            sig = inspect.signature(obj)
        defaults = list(args.defaults)
        kw_defaults = list(args.kw_defaults)
        parameters = list(sig.parameters.values())
        for (i, param) in enumerate(parameters):
            if param.default is param.empty:
                if param.kind == param.KEYWORD_ONLY:
                    kw_defaults.pop(0)
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                default = defaults.pop(0)
                value = get_default_value(lines, default)
                if value is None:
                    value = ast_unparse(default)
                parameters[i] = param.replace(default=DefaultValue(value))
            else:
                default = kw_defaults.pop(0)
                value = get_default_value(lines, default)
                if value is None:
                    value = ast_unparse(default)
                parameters[i] = param.replace(default=DefaultValue(value))
        sig = sig.replace(parameters=parameters)
        try:
            obj.__signature__ = sig
        except AttributeError:
            obj.__dict__['__signature__'] = sig
    except (AttributeError, TypeError):
        return
    except NotImplementedError as exc:
        logger.warning(__('Failed to parse a default argument value for %r: %s'), obj, exc)

def setup(app: Sphinx) -> dict[str, Any]:
    if False:
        print('Hello World!')
    app.add_config_value('autodoc_preserve_defaults', False, True)
    app.connect('autodoc-before-process-signature', update_defvalue)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}