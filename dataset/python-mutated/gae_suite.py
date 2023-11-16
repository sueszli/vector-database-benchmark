"""Oppia test suite.

In general, this script should not be run directly. Instead, invoke
it from the command line by running

    python -m scripts.run_backend_tests

from the oppia/ root folder.
"""
from __future__ import annotations
import argparse
import os
import sys
import unittest
from typing import Final, List, Optional
sys.path.insert(1, os.getcwd())
from scripts import common
CURR_DIR: Final = os.path.abspath(os.getcwd())
OPPIA_TOOLS_DIR: Final = os.path.join(CURR_DIR, '..', 'oppia_tools')
THIRD_PARTY_DIR: Final = os.path.join(CURR_DIR, 'third_party')
THIRD_PARTY_PYTHON_LIBS_DIR: Final = os.path.join(THIRD_PARTY_DIR, 'python_libs')
GOOGLE_APP_ENGINE_SDK_HOME: Final = os.path.join(OPPIA_TOOLS_DIR, 'google-cloud-sdk-335.0.0', 'google-cloud-sdk', 'platform', 'google_appengine')
_PARSER: Final = argparse.ArgumentParser()
_PARSER.add_argument('--test_target', help='optional dotted module name of the test(s) to run', type=str)

def create_test_suites(test_target: Optional[str]=None) -> List[unittest.TestSuite]:
    if False:
        return 10
    'Creates test suites. If test_target is None, runs all tests.\n\n    Args:\n        test_target: str. The name of the test script.\n            Default to None if not specified.\n\n    Returns:\n        list. A list of tests within the test script.\n\n    Raises:\n        Exception. The delimeter in the test_target should be a dot (.)\n    '
    if test_target and '/' in test_target:
        raise Exception('The delimiter in test_target should be a dot (.)')
    loader = unittest.TestLoader()
    master_test_suite = loader.loadTestsFromName(test_target) if test_target else loader.discover(CURR_DIR, pattern='[^core/tests/data]*_test.py', top_level_dir=CURR_DIR)
    return [master_test_suite]

def main(args: Optional[List[str]]=None) -> None:
    if False:
        while True:
            i = 10
    'Runs the tests.\n\n    Args:\n        args: list. A list of arguments to parse.\n\n    Raises:\n        Exception. Directory invalid_path does not exist.\n    '
    parsed_args = _PARSER.parse_args(args=args)
    for directory in common.DIRS_TO_ADD_TO_SYS_PATH:
        if not os.path.exists(os.path.dirname(directory)):
            raise Exception('Directory %s does not exist.' % directory)
        sys.path.insert(0, directory)
    sys.path = [path for path in sys.path if 'coverage' not in path]
    import dev_appserver
    dev_appserver.fix_sys_path()
    google_path = os.path.join(THIRD_PARTY_PYTHON_LIBS_DIR, 'google')
    google_module = sys.modules['google']
    google_module.__path__ = [google_path, THIRD_PARTY_PYTHON_LIBS_DIR]
    google_module.__file__ = os.path.join(google_path, '__init__.py')
    suites = create_test_suites(test_target=parsed_args.test_target)
    results = [unittest.TextTestRunner(verbosity=2).run(suite) for suite in suites]
    for result in results:
        if result.errors or result.failures:
            raise Exception('Test suite failed: %s tests run, %s errors, %s failures.' % (result.testsRun, len(result.errors), len(result.failures)))
if __name__ == '__main__':
    main()