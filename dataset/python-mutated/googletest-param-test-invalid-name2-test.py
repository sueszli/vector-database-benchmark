"""Verifies that Google Test warns the user when not initialized properly."""
import gtest_test_utils
binary_name = 'googletest-param-test-invalid-name2-test_'
COMMAND = gtest_test_utils.GetTestExecutablePath(binary_name)

def Assert(condition):
    if False:
        for i in range(10):
            print('nop')
    if not condition:
        raise AssertionError

def TestExitCodeAndOutput(command):
    if False:
        while True:
            i = 10
    'Runs the given command and verifies its exit code and output.'
    err = "Duplicate parameterized test name 'a'"
    p = gtest_test_utils.Subprocess(command)
    Assert(p.terminated_by_signal)
    Assert(err in p.output)

class GTestParamTestInvalidName2Test(gtest_test_utils.TestCase):

    def testExitCodeAndOutput(self):
        if False:
            return 10
        TestExitCodeAndOutput(COMMAND)
if __name__ == '__main__':
    gtest_test_utils.Main()