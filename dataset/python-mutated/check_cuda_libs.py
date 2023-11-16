"""Verifies that a list of libraries is installed on the system.

Takes a list of arguments with every two subsequent arguments being a logical
tuple of (path, check_soname). The path to the library and either True or False
to indicate whether to check the soname field on the shared library.

Example Usage:
./check_cuda_libs.py /path/to/lib1.so True /path/to/lib2.so False
"""
import os
import os.path
import platform
import subprocess
import sys
try:
    from shutil import which
except ImportError:
    from distutils.spawn import find_executable as which

class ConfigError(Exception):
    pass

def _is_windows():
    if False:
        for i in range(10):
            print('nop')
    return platform.system() == 'Windows'

def check_cuda_lib(path, check_soname=True):
    if False:
        while True:
            i = 10
    'Tests if a library exists on disk and whether its soname matches the filename.\n\n  Args:\n    path: the path to the library.\n    check_soname: whether to check the soname as well.\n\n  Raises:\n    ConfigError: If the library does not exist or if its soname does not match\n    the filename.\n  '
    if not os.path.isfile(path):
        raise ConfigError('No library found under: ' + path)
    objdump = which('objdump')
    if check_soname and objdump is not None and (not _is_windows()):
        output = subprocess.check_output([objdump, '-p', path]).decode('utf-8')
        output = [line for line in output.splitlines() if 'SONAME' in line]
        sonames = [line.strip().split(' ')[-1] for line in output]
        if not any((soname == os.path.basename(path) for soname in sonames)):
            raise ConfigError('None of the libraries match their SONAME: ' + path)

def main():
    if False:
        return 10
    try:
        args = [argv for argv in sys.argv[1:]]
        if len(args) % 2 == 1:
            raise ConfigError('Expected even number of arguments')
        checked_paths = []
        for i in range(0, len(args), 2):
            path = args[i]
            check_cuda_lib(path, check_soname=args[i + 1] == 'True')
            checked_paths.append(path)
        print(os.linesep.join(checked_paths))
    except ConfigError as e:
        sys.stderr.write(str(e))
        sys.exit(1)
if __name__ == '__main__':
    main()