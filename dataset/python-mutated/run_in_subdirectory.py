import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Tuple
if __name__ not in ('__main__', '__mp_main__'):
    raise SystemExit('This file is intended to be executed as an executable program. You cannot use it as a module.To run this script, run the ./{__file__} command')

def is_relative_to(path: Path, *other):
    if False:
        print('Hello World!')
    'Return True if the path is relative to another path or False.\n\n    This function is backported from Python 3.9 - Path.relativeto.\n    '
    try:
        path.relative_to(*other)
        return True
    except ValueError:
        return False

def display_usage():
    if False:
        return 10
    prog = Path(__file__).name
    print(textwrap.dedent(f'    usage: {prog} [-h] SUBDIRECTORY ARGS [ARGS ...]\n\n    Runs the program in a subdirectory and fix paths in arguments.\n\n    example:\n\n    When this program is executed with the following command:\n       {prog} frontend/ yarn eslint frontend/src/index.ts\n    Then the command will be executed:\n        yarn eslint src/index.ts\n    and the current working directory will be set to frontend/\n\n    positional arguments:\n      SUBDIRECTORY  subdirectory within which the subprocess will be executed\n      ARGS  sequence of program arguments\n\n    optional arguments:\n      -h, --help    show this help message and exit    '))

def parse_args() -> Tuple[str, List[str]]:
    if False:
        i = 10
        return i + 15
    if len(sys.argv) == 2 and sys.argv[1] in ('-h', '--help'):
        display_usage()
        sys.exit(0)
    if len(sys.argv) < 3:
        print('Missing arguments')
        display_usage()
        sys.exit(1)
    print(sys.argv)
    return (sys.argv[1], sys.argv[2:])

def fix_arg(subdirectory: str, arg: str) -> str:
    if False:
        i = 10
        return i + 15
    arg_path = Path(arg)
    if not (arg_path.exists() and is_relative_to(arg_path, subdirectory)):
        return arg
    return str(arg_path.relative_to(subdirectory))

def main():
    if False:
        for i in range(10):
            print('nop')
    (subdirectory, subprocess_args) = parse_args()
    fixed_args = [fix_arg(subdirectory, arg) for arg in subprocess_args]
    try:
        subprocess.run(fixed_args, cwd=subdirectory, check=True)
    except subprocess.CalledProcessError as ex:
        sys.exit(ex.returncode)
main()