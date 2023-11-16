"""Unit test for Google Test's global test environment behavior.

A user can specify a global test environment via
testing::AddGlobalTestEnvironment. Failures in the global environment should
result in all unit tests being skipped.

This script tests such functionality by invoking
googletest-global-environment-unittest_ (a program written with Google Test).
"""
import re
from googletest.test import gtest_test_utils

def RunAndReturnOutput(args=None):
    if False:
        while True:
            i = 10
    'Runs the test program and returns its output.'
    return gtest_test_utils.Subprocess([gtest_test_utils.GetTestExecutablePath('googletest-global-environment-unittest_')] + (args or [])).output

class GTestGlobalEnvironmentUnitTest(gtest_test_utils.TestCase):
    """Tests global test environment failures."""

    def testEnvironmentSetUpFails(self):
        if False:
            print('Hello World!')
        'Tests the behavior of not specifying the fail_fast.'
        txt = RunAndReturnOutput()
        self.assertIn('Canned environment setup error', txt)
        self.assertIn('[  SKIPPED ] 1 test', txt)
        self.assertIn('[  PASSED  ] 0 tests', txt)
        self.assertNotIn('Unexpected call', txt)

    def testEnvironmentSetUpAndTornDownForEachRepeat(self):
        if False:
            i = 10
            return i + 15
        'Tests the behavior of test environments and gtest_repeat.'
        txt = RunAndReturnOutput(['--gtest_repeat=2', '--gtest_recreate_environments_when_repeating=true'])
        expected_pattern = '(.|\n)*Repeating all tests \\(iteration 1\\)(.|\n)*Global test environment set-up.(.|\n)*SomeTest.DoesFoo(.|\n)*Global test environment tear-down(.|\n)*Repeating all tests \\(iteration 2\\)(.|\n)*Global test environment set-up.(.|\n)*SomeTest.DoesFoo(.|\n)*Global test environment tear-down(.|\n)*'
        self.assertRegex(txt, expected_pattern)

    def testEnvironmentSetUpAndTornDownOnce(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests environment and --gtest_recreate_environments_when_repeating.'
        txt = RunAndReturnOutput(['--gtest_repeat=2'])
        expected_pattern = '(.|\n)*Repeating all tests \\(iteration 1\\)(.|\n)*Global test environment set-up.(.|\n)*SomeTest.DoesFoo(.|\n)*Repeating all tests \\(iteration 2\\)(.|\n)*SomeTest.DoesFoo(.|\n)*Global test environment tear-down(.|\n)*'
        self.assertRegex(txt, expected_pattern)
        self.assertEqual(len(re.findall('Global test environment set-up', txt)), 1)
        self.assertEqual(len(re.findall('Global test environment tear-down', txt)), 1)
if __name__ == '__main__':
    gtest_test_utils.Main()