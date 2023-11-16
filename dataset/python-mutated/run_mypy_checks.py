"""MyPy test runner script."""
from __future__ import annotations
import argparse
import os
import site
import subprocess
import sys
from core import feconf
from scripts import common
from scripts import install_third_party_libs
from typing import Final, List, Optional, Tuple
EXCLUDED_DIRECTORIES: Final = ['proto_files/', 'scripts/linters/test_files/', 'third_party/', 'venv/', 'core/tests/build_sources/', 'core/tests/data/']
CONFIG_FILE_PATH: Final = os.path.join('.', 'mypy.ini')
MYPY_REQUIREMENTS_FILE_PATH: Final = os.path.join('.', 'mypy_requirements.txt')
MYPY_TOOLS_DIR: Final = os.path.join(os.getcwd(), 'third_party', 'python3_libs')
PYTHON3_CMD: Final = 'python3'
_PATHS_TO_INSERT: Final = [MYPY_TOOLS_DIR]
_PARSER: Final = argparse.ArgumentParser(description='Python type checking using mypy script.')
_PARSER.add_argument('--skip-install', help='If passed, skips installing dependencies. By default, they are installed.', action='store_true')
_PARSER.add_argument('--install-globally', help='optional; if specified, installs mypy and its requirements globally. By default, they are installed to %s' % MYPY_TOOLS_DIR, action='store_true')
_PARSER.add_argument('--files', help='Files to type-check', action='store', nargs='+')

def install_third_party_libraries(skip_install: bool) -> None:
    if False:
        print('Hello World!')
    'Run the installation script.\n\n    Args:\n        skip_install: bool. Whether to skip running the installation script.\n    '
    if not skip_install:
        install_third_party_libs.main()

def get_mypy_cmd(files: Optional[List[str]], mypy_exec_path: str, using_global_mypy: bool) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Return the appropriate command to be run.\n\n    Args:\n        files: Optional[List[str]]. List of files provided to check for MyPy\n            type checking, or None if no file is provided explicitly.\n        mypy_exec_path: str. Path of mypy executable.\n        using_global_mypy: bool. Whether generated command should run using\n            global mypy.\n\n    Returns:\n        list(str). List of command line arguments.\n    '
    if using_global_mypy:
        mypy_cmd = 'mypy'
    else:
        mypy_cmd = mypy_exec_path
    if files:
        cmd = [mypy_cmd, '--config-file', CONFIG_FILE_PATH] + files
    else:
        excluded_files_regex = '|'.join(EXCLUDED_DIRECTORIES)
        cmd = [mypy_cmd, '--exclude', excluded_files_regex, '--config-file', CONFIG_FILE_PATH, '.']
    return cmd

def install_mypy_prerequisites(install_globally: bool) -> Tuple[int, str]:
    if False:
        i = 10
        return i + 15
    'Install mypy and type stubs from mypy_requirements.txt.\n\n    Args:\n        install_globally: bool. Whether mypy and its requirements are to be\n            installed globally.\n\n    Returns:\n        tuple(int, str). The return code from installing prerequisites and the\n        path of the mypy executable.\n\n    Raises:\n        Exception. No USER_BASE found for the user.\n    '
    if install_globally:
        cmd = [PYTHON3_CMD, '-m', 'pip', 'install', '-r', MYPY_REQUIREMENTS_FILE_PATH]
    else:
        cmd = [PYTHON3_CMD, '-m', 'pip', 'install', '-r', MYPY_REQUIREMENTS_FILE_PATH, '--target', MYPY_TOOLS_DIR, '--upgrade']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.communicate()
    if b"can't combine user with prefix" in output[1]:
        uextention_text = ['--user', '--prefix=', '--system']
        new_process = subprocess.Popen(cmd + uextention_text, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        new_process.communicate()
        if site.USER_BASE is None:
            raise Exception('No USER_BASE found for the user.')
        _PATHS_TO_INSERT.append(os.path.join(site.USER_BASE, 'bin'))
        mypy_exec_path = os.path.join(site.USER_BASE, 'bin', 'mypy')
        return (new_process.returncode, mypy_exec_path)
    else:
        _PATHS_TO_INSERT.append(os.path.join(MYPY_TOOLS_DIR, 'bin'))
        mypy_exec_path = os.path.join(MYPY_TOOLS_DIR, 'bin', 'mypy')
        return (process.returncode, mypy_exec_path)

def main(args: Optional[List[str]]=None) -> int:
    if False:
        return 10
    'Runs the MyPy type checks.'
    parsed_args = _PARSER.parse_args(args=args)
    for directory in common.DIRS_TO_ADD_TO_SYS_PATH:
        sys.path.insert(1, directory)
    if not feconf.OPPIA_IS_DOCKERIZED:
        install_third_party_libraries(parsed_args.skip_install)
        print('Installing Mypy and stubs for third party libraries.')
        (return_code, mypy_exec_path) = install_mypy_prerequisites(parsed_args.install_globally)
        if return_code != 0:
            print('Cannot install Mypy and stubs for third party libraries.')
            sys.exit(1)
        print('Installed Mypy and stubs for third party libraries.')
    mypy_exec_path = os.path.join(MYPY_TOOLS_DIR, 'bin', 'mypy')
    print('Starting Mypy type checks.')
    cmd = get_mypy_cmd(parsed_args.files, mypy_exec_path, parsed_args.install_globally)
    env = os.environ.copy()
    for path in _PATHS_TO_INSERT:
        env['PATH'] = '%s%s' % (path, os.pathsep) + env['PATH']
    env['PYTHONPATH'] = MYPY_TOOLS_DIR
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    (stdout, stderr) = process.communicate()
    print(stdout.decode('utf-8'))
    print(stderr.decode('utf-8'))
    if process.returncode == 0:
        print('Mypy type checks successful.')
    else:
        print('Mypy type checks unsuccessful. Please fix the errors. For more information, visit: https://github.com/oppia/oppia/wiki/Backend-Type-Annotations')
        sys.exit(2)
    return process.returncode
if __name__ == '__main__':
    main()