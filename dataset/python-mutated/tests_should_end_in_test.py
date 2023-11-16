from __future__ import annotations
import argparse
import os.path
import re
from typing import Sequence

def main(argv: Sequence[str] | None=None) -> int:
    if False:
        return 10
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*')
    mutex = parser.add_mutually_exclusive_group()
    mutex.add_argument('--pytest', dest='pattern', action='store_const', const='.*_test\\.py', default='.*_test\\.py', help='(the default) ensure tests match %(const)s')
    mutex.add_argument('--pytest-test-first', dest='pattern', action='store_const', const='test_.*\\.py', help='ensure tests match %(const)s')
    mutex.add_argument('--django', '--unittest', dest='pattern', action='store_const', const='test.*\\.py', help='ensure tests match %(const)s')
    args = parser.parse_args(argv)
    retcode = 0
    reg = re.compile(args.pattern)
    for filename in args.filenames:
        base = os.path.basename(filename)
        if not reg.fullmatch(base) and (not base == '__init__.py') and (not base == 'conftest.py'):
            retcode = 1
            print(f'{filename} does not match pattern "{args.pattern}"')
    return retcode
if __name__ == '__main__':
    raise SystemExit(main())