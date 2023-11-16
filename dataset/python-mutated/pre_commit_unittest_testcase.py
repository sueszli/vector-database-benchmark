from __future__ import annotations
import ast
import pathlib
import sys

def check_test_file(file: str) -> int:
    if False:
        return 10
    node = ast.parse(pathlib.Path(file).read_text('utf-8'), file)
    found = 0
    classes = [c for c in node.body if isinstance(c, ast.ClassDef)]
    for c in classes:
        if any((isinstance(base, ast.Attribute) and base.attr == 'TestCase' or (isinstance(base, ast.Name) and base.id == 'TestCase') for base in c.bases)):
            found += 1
            print(f'The class {c.name} inherits from TestCase, please use pytest instead')
    return found

def main(*args: str) -> int:
    if False:
        return 10
    return sum((check_test_file(file) for file in args[1:]))
if __name__ == '__main__':
    sys.exit(main(*sys.argv))