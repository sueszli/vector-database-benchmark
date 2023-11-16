"""Check that text files with a shebang are executable."""
from __future__ import annotations
import argparse
import shlex
import sys
from typing import Sequence
from pre_commit_hooks.check_executables_have_shebangs import EXECUTABLE_VALUES
from pre_commit_hooks.check_executables_have_shebangs import git_ls_files
from pre_commit_hooks.check_executables_have_shebangs import has_shebang

def check_shebangs(paths: list[str]) -> int:
    if False:
        for i in range(10):
            print('nop')
    return _check_git_filemode(paths)

def _check_git_filemode(paths: Sequence[str]) -> int:
    if False:
        while True:
            i = 10
    seen: set[str] = set()
    for ls_file in git_ls_files(paths):
        is_executable = any((b in EXECUTABLE_VALUES for b in ls_file.mode[-3:]))
        if not is_executable and has_shebang(ls_file.filename):
            _message(ls_file.filename)
            seen.add(ls_file.filename)
    return int(bool(seen))

def _message(path: str) -> None:
    if False:
        return 10
    print(f'{path}: has a shebang but is not marked executable!\n  If it is supposed to be executable, try: `chmod +x {shlex.quote(path)}`\n  If on Windows, you may also need to: `git add --chmod=+x {shlex.quote(path)}`\n  If it not supposed to be executable, double-check its shebang is wanted.\n', file=sys.stderr)

def main(argv: Sequence[str] | None=None) -> int:
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filenames', nargs='*')
    args = parser.parse_args(argv)
    return check_shebangs(args.filenames)
if __name__ == '__main__':
    raise SystemExit(main())