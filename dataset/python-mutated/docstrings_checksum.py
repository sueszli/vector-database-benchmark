from pathlib import Path
from typing import Iterator
import ast
import hashlib

def docstrings_checksum(python_files: Iterator[Path]):
    if False:
        i = 10
        return i + 15
    files_content = (f.read_text() for f in python_files)
    trees = (ast.parse(c) for c in files_content)
    docstrings = []
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Module)):
                continue
            docstring = ast.get_docstring(node)
            if docstring:
                docstrings.append(docstring)
    docstrings.sort()
    return hashlib.md5(str(docstrings).encode('utf-8')).hexdigest()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Haystack root folder', required=True, type=Path)
    args = parser.parse_args()
    root: Path = args.root.absolute()
    haystack_files = root.glob('haystack/**/*.py')
    rest_api_files = root.glob('rest_api/**/*.py')
    import itertools
    python_files = itertools.chain(haystack_files, rest_api_files)
    md5 = docstrings_checksum(python_files)
    print(md5)