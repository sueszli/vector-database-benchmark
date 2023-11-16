"""Script for running tests for custom eslint checks."""
from __future__ import annotations
import os
import re
import subprocess
import sys
from scripts import common

def main() -> None:
    if False:
        while True:
            i = 10
    'Run the tests.'
    node_path = os.path.join(common.NODE_PATH, 'bin', 'node')
    nyc_path = os.path.join('node_modules', 'nyc', 'bin', 'nyc.js')
    mocha_path = os.path.join('node_modules', 'mocha', 'bin', 'mocha')
    filepath = 'scripts/linters/custom_eslint_checks/rules/'
    proc_args = [node_path, nyc_path, mocha_path, filepath]
    proc = subprocess.Popen(proc_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (encoded_tests_stdout, encoded_tests_stderr) = proc.communicate()
    tests_stdout = encoded_tests_stdout.decode('utf-8')
    tests_stderr = encoded_tests_stderr.decode('utf-8')
    if tests_stderr:
        print(tests_stderr)
        sys.exit(1)
    print(tests_stdout)
    if 'failing' in tests_stdout:
        print('---------------------------')
        print('Tests not passed')
        print('---------------------------')
        sys.exit(1)
    else:
        print('---------------------------')
        print('All tests passed')
        print('---------------------------')
    coverage_result = re.search('All files\\s*\\|\\s*(?P<stmts>\\S+)\\s*\\|\\s*(?P<branch>\\S+)\\s*\\|\\s*(?P<funcs>\\S+)\\s*\\|\\s*(?P<lines>\\S+)\\s*\\|\\s*', tests_stdout)
    assert coverage_result is not None
    if coverage_result.group('stmts') != '100' or coverage_result.group('branch') != '100' or coverage_result.group('funcs') != '100' or (coverage_result.group('lines') != '100'):
        raise Exception('Eslint test coverage is not 100%')
if __name__ == '__main__':
    main()