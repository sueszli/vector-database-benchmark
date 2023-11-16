"""This script runs the following tests in all cases.
- Javascript and Python Linting
- Backend Python tests

Only when frontend files are changed will it run Frontend Karma unit tests.
"""
from __future__ import annotations
import argparse
import subprocess
from typing import Final, List, Optional
from . import common
from . import run_backend_tests
from . import run_frontend_tests
from .linters import pre_commit_linter
_PARSER: Final = argparse.ArgumentParser(description="\nRun this script from the oppia root folder prior to opening a PR:\n    python -m scripts.run_presubmit_checks\nSet the origin branch to compare against by adding\n--branch=your_branch or -b=your_branch\nBy default, if the current branch tip exists on remote origin,\nthe current branch is compared against its tip on GitHub.\nOtherwise it's compared against 'develop'.\nThis script runs the following tests in all cases.\n- Javascript and Python Linting\n- Backend Python tests\nOnly when frontend files are changed will it run Frontend Karma unit tests.\nIf any of these tests result in errors, this script will terminate.\nNote: The test scripts are arranged in increasing order of time taken. This\nenables a broken build to be detected as quickly as possible.\n")
_PARSER.add_argument('--branch', '-b', help='optional; if specified, the origin branch to compare against.')

def main(args: Optional[List[str]]=None) -> None:
    if False:
        print('Hello World!')
    'Run the presubmit checks.'
    parsed_args = _PARSER.parse_args(args=args)
    print('Linting files since the last commit')
    pre_commit_linter.main(args=[])
    print('Linting passed.')
    print('')
    current_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], encoding='utf-8')
    matched_branch_num = subprocess.check_output(['git', 'ls-remote', '--heads', 'origin', current_branch, '|', 'wc', '-l'], encoding='utf-8')
    if parsed_args.branch:
        branch = parsed_args.branch
    elif matched_branch_num == '1':
        branch = 'origin/%s' % current_branch
    else:
        branch = 'develop'
    print('Comparing the current branch with %s' % branch)
    all_changed_files = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM', branch], encoding='utf-8')
    if common.FRONTEND_DIR in all_changed_files:
        print('Running frontend unit tests')
        run_frontend_tests.main(args=['--run_minified_tests'])
        print('Frontend tests passed.')
    else:
        common.print_each_string_after_two_new_lines(['No frontend files were changed.', 'Skipped frontend tests'])
    print('Running backend tests')
    run_backend_tests.main(args=[])
    print('Backend tests passed.')
if __name__ == '__main__':
    main()