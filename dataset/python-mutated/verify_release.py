from __future__ import annotations
import argparse
import ast
import concurrent.futures
import itertools
import sys
from functools import partial
from pathlib import Path
from typing import NamedTuple
from packaging import version as v

class RemovedIn(NamedTuple):
    version: v.Version
    lineno: int

class RemovedInVisitor(ast.NodeVisitor):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.removed_ins: list[RemovedIn] = []

    def visit_keyword(self, node: ast.keyword) -> None:
        if False:
            return 10
        if node.arg == 'removed_in':
            value = node.value
            if isinstance(value, ast.Constant):
                self.removed_ins.append(RemovedIn(version=normalize_version(value.value), lineno=node.lineno))

def normalize_version(vstr: str) -> v.Version:
    if False:
        return 10
    ver = v.parse(vstr)
    return v.parse(f'{ver.major}.{ver.minor}.{ver.micro}')

def find_invalid_removed_ins(next_ver: v.Version, path: Path) -> list[tuple[Path, RemovedIn]]:
    if False:
        for i in range(10):
            print('nop')
    code = path.read_text()
    node = ast.parse(code)
    visitor = RemovedInVisitor()
    visitor.visit(node)
    return [(path, removed_in) for removed_in in visitor.removed_ins if removed_in.version <= next_ver]

def main(next_ver: v.Version, root: Path) -> int:
    if False:
        while True:
            i = 10
    pyfiles = root.rglob('*.py')
    find_invalid = partial(find_invalid_removed_ins, next_ver)
    with concurrent.futures.ProcessPoolExecutor() as e:
        msgs = e.map(find_invalid, pyfiles)
    infos = sorted(itertools.chain.from_iterable(msgs))
    for (path, removed_in) in infos:
        print(f'{path}:{removed_in.lineno:d} (removed_in={removed_in.version} <= next={next_ver})', file=sys.stderr)
    return len(infos)
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Verify that removed_in= matches are greater than the next version.')
    p.add_argument('next_version', help='The next release version.')
    p.add_argument('-r', '--root', type=Path, default=Path(), help='Root directory to search for Python modules. Defaults to the current directory.')
    args = p.parse_args()
    sys.exit(main(normalize_version(args.next_version), args.root))