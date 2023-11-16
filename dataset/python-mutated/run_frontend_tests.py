"""This script runs unit tests for frontend JavaScript code (using Karma)."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from scripts import common
from typing import Optional, Sequence
from . import build
from . import check_frontend_test_coverage
from . import install_third_party_libs
DTSLINT_TYPE_TESTS_DIR_RELATIVE_PATH = os.path.join('typings', 'tests')
TYPESCRIPT_DIR_RELATIVE_PATH = os.path.join('node_modules', 'typescript', 'lib')
MAX_ATTEMPTS = 2
_PARSER = argparse.ArgumentParser(description="\nRun this script from the oppia root folder:\n    python -m scripts.run_frontend_tests\nThe root folder MUST be named 'oppia'.\nNote: You can replace 'it' with 'fit' or 'describe' with 'fdescribe' to run\na single test or test suite.\n")
_PARSER.add_argument('--dtslint_only', help='optional; if specified, only runs dtslint type tests.', action='store_true')
_PARSER.add_argument('--skip_install', help='optional; if specified, skips installing dependencies', action='store_true')
_PARSER.add_argument('--verbose', help='optional; if specified, enables the karma terminal and prints all the logs.', action='store_true')
_PARSER.add_argument('--run_minified_tests', help='optional; if specified, runs frontend karma tests on both minified and non-minified code', action='store_true')
_PARSER.add_argument('--check_coverage', help='optional; if specified, checks frontend test coverage', action='store_true')
_PARSER.add_argument('--download_combined_frontend_spec_file', help='optional; if specifided, downloads the combined frontend file', action='store_true')

def run_dtslint_type_tests() -> None:
    if False:
        while True:
            i = 10
    'Runs the dtslint type tests in typings/tests.'
    print('Running dtslint type tests.')
    cmd = ['./node_modules/dtslint/bin/index.js', DTSLINT_TYPE_TESTS_DIR_RELATIVE_PATH, '--localTs', TYPESCRIPT_DIR_RELATIVE_PATH]
    task = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output_lines = []
    assert task.stdout is not None
    while True:
        line = task.stdout.readline()
        if len(line) == 0 and task.poll() is not None:
            break
        if line:
            print(line, end='')
            output_lines.append(line)
    print('Done!')
    if task.returncode:
        sys.exit('The dtslint (type tests) failed.')

def main(args: Optional[Sequence[str]]=None) -> None:
    if False:
        print('Hello World!')
    'Runs the frontend tests.'
    parsed_args = _PARSER.parse_args(args=args)
    run_dtslint_type_tests()
    if parsed_args.dtslint_only:
        return
    if not parsed_args.skip_install:
        install_third_party_libs.main()
    common.setup_chrome_bin_env_variable()
    build.save_hashes_to_file({})
    common.print_each_string_after_two_new_lines(['View interactive frontend test coverage reports by navigating to', '../karma_coverage_reports', 'on your filesystem.', 'Running test in development environment'])
    cmd = [common.NODE_BIN_PATH, '--max-old-space-size=4096', os.path.join(common.NODE_MODULES_PATH, 'karma', 'bin', 'karma'), 'start', os.path.join('core', 'tests', 'karma.conf.ts')]
    if parsed_args.run_minified_tests:
        print('Running test in production environment')
        build.main(args=['--prod_env', '--minify_third_party_libs_only'])
        cmd.append('--prodEnv')
    else:
        build.main(args=[])
    if parsed_args.verbose:
        cmd.append('--terminalEnabled')
    for attempt in range(MAX_ATTEMPTS):
        print(f'Attempt {attempt + 1} of {MAX_ATTEMPTS}')
        task = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output_lines = []
        assert task.stdout is not None
        combined_spec_file_started_downloading = False
        download_task = None
        while True:
            line = task.stdout.readline()
            if len(line) == 0 and task.poll() is not None:
                break
            if line and (not '[web-server]:' in line.decode('utf-8')):
                print(line.decode('utf-8'), end='')
                output_lines.append(line)
            if 'Executed' in line.decode('utf-8') and (not combined_spec_file_started_downloading) and parsed_args.download_combined_frontend_spec_file:
                download_task = subprocess.Popen(['wget', 'http://localhost:9876/base/core/templates/' + 'combined-tests.spec.js', '-P', os.path.join('../karma_coverage_reports')])
                download_task.wait()
                combined_spec_file_started_downloading = True
        concatenated_output = ''.join((line.decode('utf-8') for line in output_lines))
        if download_task:
            if download_task.returncode:
                print('Failed to download the combined-tests.spec.js file.')
            else:
                print('Downloaded the combined-tests.spec.js file and storedin ../karma_coverage_reports')
        print('Done!')
        if 'Trying to get the Angular injector' in concatenated_output:
            print('If you run into the error "Trying to get the Angular injector", please see https://github.com/oppia/oppia/wiki/Frontend-unit-tests-guide#how-to-handle-common-errors for details on how to fix it.')
        if 'Disconnected , because no message' in concatenated_output:
            print('Detected chrome disconnected flake (#16607), so rerunning if attempts allow.')
        else:
            break
    if parsed_args.check_coverage:
        if task.returncode:
            sys.exit('The frontend tests failed. Please fix it before running the test coverage check.')
        else:
            check_frontend_test_coverage.main()
    elif task.returncode:
        sys.exit(task.returncode)
if __name__ == '__main__':
    main()