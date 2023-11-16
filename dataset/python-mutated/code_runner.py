""" Provide a utility class ``CodeRunner`` for use by handlers that execute
Python source code.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import os
import sys
import traceback
from os.path import basename
from types import CodeType, ModuleType
from typing import Callable
from ...core.types import PathLike
from ...util.serialization import make_globally_unique_id
from .handler import handle_exception
__all__ = ('CodeRunner',)

class CodeRunner:
    """ Compile and run Python source code.

    """
    _code: CodeType | None
    _doc: str | None
    _permanent_error: str | None
    _permanent_error_detail: str | None
    _path: PathLike
    _source: str
    _argv: list[str]
    _package: ModuleType | None
    ran: bool
    _failed: bool
    _error: str | None
    _error_detail: str | None

    def __init__(self, source: str, path: PathLike, argv: list[str], package: ModuleType | None=None) -> None:
        if False:
            return 10
        '\n\n        Args:\n            source (str) :\n                A string containing Python source code to execute\n\n            path (str) :\n                A filename to use in any debugging or error output\n\n            argv (list[str]) :\n                A list of string arguments to make available as ``sys.argv``\n                when the code executes\n\n            package (bool) :\n                An optional package module to configure\n\n        Raises:\n            ValueError, if package is specified for an __init__.py\n\n        '
        if package and basename(path) == '__init__.py':
            raise ValueError('__init__.py cannot have package specified')
        self._permanent_error = None
        self._permanent_error_detail = None
        self.reset_run_errors()
        import ast
        self._code = None
        try:
            nodes = ast.parse(source, os.fspath(path))
            self._code = compile(nodes, filename=path, mode='exec', dont_inherit=True)
            d = dict(zip(self._code.co_names, self._code.co_consts))
            self._doc = d.get('__doc__', None)
        except SyntaxError as e:
            self._code = None
            filename = os.path.basename(e.filename) if e.filename is not None else '???'
            self._permanent_error = f"Invalid syntax in {filename!r} on line {e.lineno or '???'}:\n{e.text or '???'}"
            self._permanent_error_detail = traceback.format_exc()
        self._path = path
        self._source = source
        self._argv = argv
        self._package = package
        self.ran = False

    @property
    def doc(self) -> str | None:
        if False:
            print('Hello World!')
        ' Contents of docstring, if code contains one.\n\n        '
        return self._doc

    @property
    def error(self) -> str | None:
        if False:
            return 10
        ' If code execution fails, may contain a related error message.\n\n        '
        return self._error if self._permanent_error is None else self._permanent_error

    @property
    def error_detail(self) -> str | None:
        if False:
            return 10
        ' If code execution fails, may contain a traceback or other details.\n\n        '
        return self._error_detail if self._permanent_error_detail is None else self._permanent_error_detail

    @property
    def failed(self) -> bool:
        if False:
            while True:
                i = 10
        ' ``True`` if code execution failed\n\n        '
        return self._failed or self._code is None

    @property
    def path(self) -> PathLike:
        if False:
            i = 10
            return i + 15
        ' The path that new modules will be configured with.\n\n        '
        return self._path

    @property
    def source(self) -> str:
        if False:
            print('Hello World!')
        ' The configured source code that will be executed when ``run`` is\n        called.\n\n        '
        return self._source

    def new_module(self) -> ModuleType | None:
        if False:
            return 10
        ' Make a fresh module to run in.\n\n        Returns:\n            Module\n\n        '
        self.reset_run_errors()
        if self._code is None:
            return None
        module_name = 'bokeh_app_' + make_globally_unique_id().replace('-', '')
        module = ModuleType(module_name)
        module.__dict__['__file__'] = os.path.abspath(self._path)
        if self._package:
            module.__package__ = self._package.__name__
            module.__path__ = [os.path.dirname(self._path)]
        if basename(self.path) == '__init__.py':
            module.__package__ = module_name
            module.__path__ = [os.path.dirname(self._path)]
        return module

    def reset_run_errors(self) -> None:
        if False:
            return 10
        ' Clears any transient error conditions from a previous run.\n\n        Returns\n            None\n\n        '
        self._failed = False
        self._error = None
        self._error_detail = None

    def run(self, module: ModuleType, post_check: Callable[[], None] | None=None) -> None:
        if False:
            return 10
        ' Execute the configured source code in a module and run any post\n        checks.\n\n        Args:\n            module (Module) :\n                A module to execute the configured code in.\n\n            post_check (callable, optional) :\n                A function that raises an exception if expected post-conditions\n                are not met after code execution.\n\n        '
        _cwd = os.getcwd()
        _sys_path = list(sys.path)
        _sys_argv = list(sys.argv)
        sys.path.insert(0, os.path.dirname(self._path))
        sys.argv = [os.path.basename(self._path), *self._argv]
        assert self._code is not None
        try:
            exec(self._code, module.__dict__)
            if post_check:
                post_check()
        except Exception as e:
            handle_exception(self, e)
        finally:
            os.chdir(_cwd)
            sys.path = _sys_path
            sys.argv = _sys_argv
            self.ran = True