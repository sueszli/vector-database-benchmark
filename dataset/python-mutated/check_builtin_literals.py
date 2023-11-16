from __future__ import annotations
import argparse
import ast
from typing import NamedTuple
from typing import Sequence
BUILTIN_TYPES = {'complex': '0j', 'dict': '{}', 'float': '0.0', 'int': '0', 'list': '[]', 'str': "''", 'tuple': '()'}

class Call(NamedTuple):
    name: str
    line: int
    column: int

class Visitor(ast.NodeVisitor):

    def __init__(self, ignore: Sequence[str] | None=None, allow_dict_kwargs: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        self.builtin_type_calls: list[Call] = []
        self.ignore = set(ignore) if ignore else set()
        self.allow_dict_kwargs = allow_dict_kwargs

    def _check_dict_call(self, node: ast.Call) -> bool:
        if False:
            return 10
        return self.allow_dict_kwargs and bool(node.keywords)

    def visit_Call(self, node: ast.Call) -> None:
        if False:
            i = 10
            return i + 15
        if not isinstance(node.func, ast.Name):
            return
        if node.func.id not in set(BUILTIN_TYPES).difference(self.ignore):
            return
        if node.func.id == 'dict' and self._check_dict_call(node):
            return
        elif node.args:
            return
        self.builtin_type_calls.append(Call(node.func.id, node.lineno, node.col_offset))

def check_file(filename: str, ignore: Sequence[str] | None=None, allow_dict_kwargs: bool=True) -> list[Call]:
    if False:
        print('Hello World!')
    with open(filename, 'rb') as f:
        tree = ast.parse(f.read(), filename=filename)
    visitor = Visitor(ignore=ignore, allow_dict_kwargs=allow_dict_kwargs)
    visitor.visit(tree)
    return visitor.builtin_type_calls

def parse_ignore(value: str) -> set[str]:
    if False:
        i = 10
        return i + 15
    return set(value.split(','))

def main(argv: Sequence[str] | None=None) -> int:
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*')
    parser.add_argument('--ignore', type=parse_ignore, default=set())
    mutex = parser.add_mutually_exclusive_group(required=False)
    mutex.add_argument('--allow-dict-kwargs', action='store_true')
    mutex.add_argument('--no-allow-dict-kwargs', dest='allow_dict_kwargs', action='store_false')
    mutex.set_defaults(allow_dict_kwargs=True)
    args = parser.parse_args(argv)
    rc = 0
    for filename in args.filenames:
        calls = check_file(filename, ignore=args.ignore, allow_dict_kwargs=args.allow_dict_kwargs)
        if calls:
            rc = rc or 1
        for call in calls:
            print(f'{filename}:{call.line}:{call.column}: replace {call.name}() with {BUILTIN_TYPES[call.name]}')
    return rc
if __name__ == '__main__':
    raise SystemExit(main())