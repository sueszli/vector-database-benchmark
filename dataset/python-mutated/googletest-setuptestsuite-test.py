"""Verifies that SetUpTestSuite and TearDownTestSuite errors are noticed."""
from googletest.test import gtest_test_utils
COMMAND = gtest_test_utils.GetTestExecutablePath('googletest-setuptestsuite-test_')

class GTestSetUpTestSuiteTest(gtest_test_utils.TestCase):

    def testSetupErrorAndTearDownError(self):
        if False:
            return 10
        p = gtest_test_utils.Subprocess(COMMAND)
        self.assertNotEqual(p.exit_code, 0, msg=p.output)
        self.assertIn('[  FAILED  ] SetupFailTest: SetUpTestSuite or TearDownTestSuite\n[  FAILED  ] TearDownFailTest: SetUpTestSuite or TearDownTestSuite\n\n 2 FAILED TEST SUITES\n', p.output)
if __name__ == '__main__':
    gtest_test_utils.Main()