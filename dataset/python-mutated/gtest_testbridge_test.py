"""Verifies that Google Test uses filter provided via testbridge."""
import os
import gtest_test_utils
binary_name = 'gtest_testbridge_test_'
COMMAND = gtest_test_utils.GetTestExecutablePath(binary_name)
TESTBRIDGE_NAME = 'TESTBRIDGE_TEST_ONLY'

def Assert(condition):
    if False:
        print('Hello World!')
    if not condition:
        raise AssertionError

class GTestTestFilterTest(gtest_test_utils.TestCase):

    def testTestExecutionIsFiltered(self):
        if False:
            i = 10
            return i + 15
        'Tests that the test filter is picked up from the testbridge env var.'
        subprocess_env = os.environ.copy()
        subprocess_env[TESTBRIDGE_NAME] = '*.TestThatSucceeds'
        p = gtest_test_utils.Subprocess(COMMAND, env=subprocess_env)
        self.assertEquals(0, p.exit_code)
        Assert('filter = *.TestThatSucceeds' in p.output)
        Assert('[       OK ] TestFilterTest.TestThatSucceeds' in p.output)
        Assert('[  PASSED  ] 1 test.' in p.output)
if __name__ == '__main__':
    gtest_test_utils.Main()