"""Run static analysis on the project."""
import argparse
import sys
from subprocess import CalledProcessError, check_call
from tempfile import TemporaryDirectory

def do_process(args, shell=False):
    if False:
        print('Hello World!')
    'Run program provided by args.\n\n    Return ``True`` on success.\n\n    Output failed message on non-zero exit and return False.\n\n    Exit if command is not found.\n\n    '
    print(f"Running: {' '.join(args)}")
    try:
        check_call(args, shell=shell)
    except CalledProcessError:
        print(f"\nFailed: {' '.join(args)}")
        return False
    except Exception as exc:
        sys.stderr.write(f'{str(exc)}\n')
        sys.exit(1)
    return True

def run_static():
    if False:
        i = 10
        return i + 15
    'Runs the static tests.\n\n    Returns a statuscode of 0 if everything ran correctly. Otherwise, it will return\n    statuscode 1\n\n    '
    success = True
    success &= do_process(['pre-commit', 'run', '--all-files'])
    with TemporaryDirectory() as tmp_dir:
        success &= do_process(['sphinx-build', '-W', '--keep-going', 'docs', tmp_dir])
    return success

def run_unit():
    if False:
        for i in range(10):
            print('nop')
    'Runs the unit-tests.\n\n    Follows the behavior of the static tests, where any failed tests cause pre_push.py\n    to fail.\n\n    '
    return do_process(['pytest'])

def main():
    if False:
        for i in range(10):
            print('nop')
    'Runs the main function.\n\n    usage: pre_push.py [-h] [-n] [-u] [-a]\n\n    Run static and/or unit-tests\n\n    '
    parser = argparse.ArgumentParser(description='Run static and/or unit-tests')
    parser.add_argument('-n', '--unstatic', action='store_true', help='Do not run static tests (black/flake8/pydocstyle/sphinx-build)', default=False)
    parser.add_argument('-u', '--unit-tests', '--unit', action='store_true', default=False, help='Run the unit tests')
    parser.add_argument('-a', '--all', action='store_true', default=False, help='Run all the tests (static and unit). Overrides the unstatic argument.')
    args = parser.parse_args()
    success = True
    try:
        if not args.unstatic or args.all:
            success &= run_static()
        if args.all or args.unit_tests:
            success &= run_unit()
    except KeyboardInterrupt:
        return int(not False)
    return int(not success)
if __name__ == '__main__':
    exit_code = main()
    print('\npre_push.py: Success!' if not exit_code else '\npre_push.py: Fail')
    sys.exit(exit_code)