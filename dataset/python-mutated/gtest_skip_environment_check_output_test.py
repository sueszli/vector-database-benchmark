"""Tests Google Test's gtest skip in environment setup  behavior.

This script invokes gtest_skip_in_environment_setup_test_ and verifies its
output.
"""
import gtest_test_utils
EXE_PATH = gtest_test_utils.GetTestExecutablePath('gtest_skip_in_environment_setup_test')
OUTPUT = gtest_test_utils.Subprocess([EXE_PATH]).output

class SkipEntireEnvironmentTest(gtest_test_utils.TestCase):

    def testSkipEntireEnvironmentTest(self):
        if False:
            print('Hello World!')
        self.assertIn('Skipping the entire environment', OUTPUT)
        self.assertNotIn('FAILED', OUTPUT)
if __name__ == '__main__':
    gtest_test_utils.Main()