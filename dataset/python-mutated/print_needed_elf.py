import sys
import os
import subprocess
from typing import Any, Iterable, List

def eprint(msg: Any):
    if False:
        while True:
            i = 10
    print(msg, file=sys.stderr)

def run(args: List[str]) -> str:
    if False:
        while True:
            i = 10
    result = subprocess.run(args, capture_output=True)
    if result.returncode != 0:
        eprint(result.stderr)
        eprint('Command failed with return code {}: {}'.format(result.returncode, args))
        sys.exit(result.returncode)
    return result.stdout.decode('utf-8')

def stripped_strs(strs: Iterable[str]) -> Iterable[str]:
    if False:
        for i in range(10):
            print('nop')
    return (cleaned for x in strs for cleaned in [x.strip()] if cleaned != '')

def get_rpaths(exe: str) -> Iterable[str]:
    if False:
        i = 10
        return i + 15
    return stripped_strs(run(['patchelf', '--print-rpath', exe]).split(':'))

def resolve_origin(origin: str, paths: Iterable[str]) -> Iterable[str]:
    if False:
        while True:
            i = 10
    return (path.replace('$ORIGIN', origin) for path in paths)

def get_needed(exe: str) -> Iterable[str]:
    if False:
        i = 10
        return i + 15
    return stripped_strs(run(['patchelf', '--print-needed', exe]).splitlines())

def resolve_paths(needed: Iterable[str], rpaths: List[str]) -> Iterable[str]:
    if False:
        print('Hello World!')
    existing_paths = lambda lib, paths: (abs_path for path in paths for abs_path in [os.path.join(path, lib)] if os.path.exists(abs_path))
    return (found if found is not None else eprint("Warning: can't find {} in {}".format(lib, rpaths)) for lib in needed for found in [next(existing_paths(lib, rpaths), None)])

def main(exe: str):
    if False:
        i = 10
        return i + 15
    dirname = os.path.dirname(exe)
    rpaths_raw = list(get_rpaths(exe))
    rpaths_raw = [dirname] if rpaths_raw == [] else rpaths_raw
    rpaths = list(resolve_origin(dirname, rpaths_raw))
    for path in (x for x in resolve_paths(get_needed(exe), rpaths) if x is not None):
        print(path)
if __name__ == '__main__':
    main(*sys.argv[1:])