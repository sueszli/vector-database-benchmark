"""Verifies that Google Test warns the user when not initialized properly."""
import gtest_test_utils
COMMAND = gtest_test_utils.GetTestExecutablePath('googletest-uninitialized-test_')

def Assert(condition):
    if False:
        while True:
            i = 10
    if not condition:
        raise AssertionError

def AssertEq(expected, actual):
    if False:
        for i in range(10):
            print('nop')
    if expected != actual:
        print('Expected: %s' % (expected,))
        print('  Actual: %s' % (actual,))
        raise AssertionError

def TestExitCodeAndOutput(command):
    if False:
        print('Hello World!')
    'Runs the given command and verifies its exit code and output.'
    p = gtest_test_utils.Subprocess(command)
    if p.exited and p.exit_code == 0:
        Assert('IMPORTANT NOTICE' in p.output)
    Assert('InitGoogleTest' in p.output)

class GTestUninitializedTest(gtest_test_utils.TestCase):

    def testExitCodeAndOutput(self):
        if False:
            print('Hello World!')
        TestExitCodeAndOutput(COMMAND)
if __name__ == '__main__':
    gtest_test_utils.Main()