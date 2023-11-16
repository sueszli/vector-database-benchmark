from __future__ import annotations
import ast
import itertools
import pathlib
import sys
from typing import Iterator

def iter_decorated_operators(source: pathlib.Path) -> Iterator[ast.ClassDef]:
    if False:
        for i in range(10):
            print('nop')
    mod = ast.parse(source.read_text('utf-8'), str(source))
    for node in ast.walk(mod):
        if isinstance(node, ast.ClassDef) and any((isinstance(base, ast.Name) and base.id == 'DecoratedOperator' for base in node.bases)):
            yield node

def check_missing_custom_operator_name(klass: ast.ClassDef) -> bool:
    if False:
        for i in range(10):
            print('nop')
    for node in ast.iter_child_nodes(klass):
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == 'custom_operator_name':
                return True
        elif isinstance(node, ast.Assign):
            if any((isinstance(t, ast.Name) and t.id == 'custom_operator_name' for t in node.targets)):
                return True
    return False

def main(*args: str) -> int:
    if False:
        i = 10
        return i + 15
    classes = itertools.chain.from_iterable((iter_decorated_operators(pathlib.Path(a)) for a in args[1:]))
    results = ((k.name, check_missing_custom_operator_name(k)) for k in classes)
    failures = [name for (name, success) in results if not success]
    for failure in failures:
        print(f'Missing custom_operator_name in class: {failure}')
    return len(failures)
if __name__ == '__main__':
    sys.exit(main(*sys.argv))