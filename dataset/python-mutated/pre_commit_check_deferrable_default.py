from __future__ import annotations
import ast
import glob
import itertools
import os
import sys
from typing import Iterator
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DEFERRABLE_DOC = 'https://github.com/apache/airflow/blob/main/docs/apache-airflow/authoring-and-scheduling/deferring.rst#writing-deferrable-operators'

def _is_valid_deferrable_default(default: ast.AST) -> bool:
    if False:
        return 10
    'Check whether default is \'conf.getboolean("operators", "default_deferrable", fallback=False)\''
    if not isinstance(default, ast.Call):
        return False
    call_to_conf_getboolean = isinstance(default.func, ast.Attribute) and isinstance(default.func.value, ast.Name) and (default.func.value.id == 'conf') and (default.func.attr == 'getboolean')
    if not call_to_conf_getboolean:
        return False
    return len(default.args) == 2 and isinstance(default.args[0], ast.Constant) and (default.args[0].value == 'operators') and isinstance(default.args[1], ast.Constant) and (default.args[1].value == 'default_deferrable') and (len(default.keywords) == 1) and (default.keywords[0].arg == 'fallback') and isinstance(default.keywords[0].value, ast.Constant) and (default.keywords[0].value.value is False)

def iter_check_deferrable_default_errors(module_filename: str) -> Iterator[str]:
    if False:
        while True:
            i = 10
    ast_obj = ast.parse(open(module_filename).read())
    cls_nodes = (node for node in ast.iter_child_nodes(ast_obj) if isinstance(node, ast.ClassDef))
    init_method_nodes = (node for cls_node in cls_nodes for node in ast.iter_child_nodes(cls_node) if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    for node in init_method_nodes:
        args = node.args
        arguments = reversed([*args.args, *args.kwonlyargs])
        defaults = reversed([*args.defaults, *args.kw_defaults])
        for (argument, default) in zip(arguments, defaults):
            if argument is None or default is None:
                continue
            if argument.arg != 'deferrable' or _is_valid_deferrable_default(default):
                continue
            yield f'{module_filename}:{default.lineno}'

def main() -> int:
    if False:
        print('Hello World!')
    modules = itertools.chain(glob.glob(f'{ROOT_DIR}/airflow/**/sensors/**.py', recursive=True), glob.glob(f'{ROOT_DIR}/airflow/**/operators/**.py', recursive=True))
    errors = [error for module in modules for error in iter_check_deferrable_default_errors(module)]
    if errors:
        print('Incorrect deferrable default values detected at:')
        for error in errors:
            print(f'  {error}')
        print(f'Please set the default value of deferrbale to "conf.getboolean("operators", "default_deferrable", fallback=False)"\nSee: {DEFERRABLE_DOC}\n')
    return len(errors)
if __name__ == '__main__':
    sys.exit(main())