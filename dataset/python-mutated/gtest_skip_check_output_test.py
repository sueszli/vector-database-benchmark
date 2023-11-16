"""Tests Google Test's gtest skip in environment setup  behavior.

This script invokes gtest_skip_in_environment_setup_test_ and verifies its
output.
"""
import re
from googletest.test import gtest_test_utils
EXE_PATH = gtest_test_utils.GetTestExecutablePath('gtest_skip_test')
OUTPUT = gtest_test_utils.Subprocess([EXE_PATH]).output

class SkipEntireEnvironmentTest(gtest_test_utils.TestCase):

    def testSkipEntireEnvironmentTest(self):
        if False:
            while True:
                i = 10
        self.assertIn('Skipped\nskipping single test\n', OUTPUT)
        skip_fixture = 'Skipped\nskipping all tests for this fixture\n'
        self.assertIsNotNone(re.search(skip_fixture + '.*' + skip_fixture, OUTPUT, flags=re.DOTALL), repr(OUTPUT))
        self.assertNotIn('FAILED', OUTPUT)
if __name__ == '__main__':
    gtest_test_utils.Main()