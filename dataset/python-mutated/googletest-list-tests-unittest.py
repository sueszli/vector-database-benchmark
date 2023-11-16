"""Unit test for Google Test's --gtest_list_tests flag.

A user can ask Google Test to list all tests by specifying the
--gtest_list_tests flag.  This script tests such functionality
by invoking googletest-list-tests-unittest_ (a program written with
Google Test) the command line flags.
"""
import re
import gtest_test_utils
LIST_TESTS_FLAG = 'gtest_list_tests'
EXE_PATH = gtest_test_utils.GetTestExecutablePath('googletest-list-tests-unittest_')
EXPECTED_OUTPUT_NO_FILTER_RE = re.compile('FooDeathTest\\.\n  Test1\nFoo\\.\n  Bar1\n  Bar2\n  DISABLED_Bar3\nAbc\\.\n  Xyz\n  Def\nFooBar\\.\n  Baz\nFooTest\\.\n  Test1\n  DISABLED_Test2\n  Test3\nTypedTest/0\\.  # TypeParam = (VeryLo{245}|class VeryLo{239})\\.\\.\\.\n  TestA\n  TestB\nTypedTest/1\\.  # TypeParam = int\\s*\\*( __ptr64)?\n  TestA\n  TestB\nTypedTest/2\\.  # TypeParam = .*MyArray<bool,\\s*42>\n  TestA\n  TestB\nMy/TypeParamTest/0\\.  # TypeParam = (VeryLo{245}|class VeryLo{239})\\.\\.\\.\n  TestA\n  TestB\nMy/TypeParamTest/1\\.  # TypeParam = int\\s*\\*( __ptr64)?\n  TestA\n  TestB\nMy/TypeParamTest/2\\.  # TypeParam = .*MyArray<bool,\\s*42>\n  TestA\n  TestB\nMyInstantiation/ValueParamTest\\.\n  TestA/0  # GetParam\\(\\) = one line\n  TestA/1  # GetParam\\(\\) = two\\\\nlines\n  TestA/2  # GetParam\\(\\) = a very\\\\nlo{241}\\.\\.\\.\n  TestB/0  # GetParam\\(\\) = one line\n  TestB/1  # GetParam\\(\\) = two\\\\nlines\n  TestB/2  # GetParam\\(\\) = a very\\\\nlo{241}\\.\\.\\.\n')
EXPECTED_OUTPUT_FILTER_FOO_RE = re.compile('FooDeathTest\\.\n  Test1\nFoo\\.\n  Bar1\n  Bar2\n  DISABLED_Bar3\nFooBar\\.\n  Baz\nFooTest\\.\n  Test1\n  DISABLED_Test2\n  Test3\n')

def Run(args):
    if False:
        for i in range(10):
            print('nop')
    'Runs googletest-list-tests-unittest_ and returns the list of tests printed.'
    return gtest_test_utils.Subprocess([EXE_PATH] + args, capture_stderr=False).output

class GTestListTestsUnitTest(gtest_test_utils.TestCase):
    """Tests using the --gtest_list_tests flag to list all tests."""

    def RunAndVerify(self, flag_value, expected_output_re, other_flag):
        if False:
            return 10
        'Runs googletest-list-tests-unittest_ and verifies that it prints\n    the correct tests.\n\n    Args:\n      flag_value:         value of the --gtest_list_tests flag;\n                          None if the flag should not be present.\n      expected_output_re: regular expression that matches the expected\n                          output after running command;\n      other_flag:         a different flag to be passed to command\n                          along with gtest_list_tests;\n                          None if the flag should not be present.\n    '
        if flag_value is None:
            flag = ''
            flag_expression = 'not set'
        elif flag_value == '0':
            flag = '--%s=0' % LIST_TESTS_FLAG
            flag_expression = '0'
        else:
            flag = '--%s' % LIST_TESTS_FLAG
            flag_expression = '1'
        args = [flag]
        if other_flag is not None:
            args += [other_flag]
        output = Run(args)
        if expected_output_re:
            self.assert_(expected_output_re.match(output), 'when %s is %s, the output of "%s" is "%s",\nwhich does not match regex "%s"' % (LIST_TESTS_FLAG, flag_expression, ' '.join(args), output, expected_output_re.pattern))
        else:
            self.assert_(not EXPECTED_OUTPUT_NO_FILTER_RE.match(output), 'when %s is %s, the output of "%s" is "%s"' % (LIST_TESTS_FLAG, flag_expression, ' '.join(args), output))

    def testDefaultBehavior(self):
        if False:
            return 10
        'Tests the behavior of the default mode.'
        self.RunAndVerify(flag_value=None, expected_output_re=None, other_flag=None)

    def testFlag(self):
        if False:
            while True:
                i = 10
        'Tests using the --gtest_list_tests flag.'
        self.RunAndVerify(flag_value='0', expected_output_re=None, other_flag=None)
        self.RunAndVerify(flag_value='1', expected_output_re=EXPECTED_OUTPUT_NO_FILTER_RE, other_flag=None)

    def testOverrideNonFilterFlags(self):
        if False:
            print('Hello World!')
        'Tests that --gtest_list_tests overrides the non-filter flags.'
        self.RunAndVerify(flag_value='1', expected_output_re=EXPECTED_OUTPUT_NO_FILTER_RE, other_flag='--gtest_break_on_failure')

    def testWithFilterFlags(self):
        if False:
            return 10
        'Tests that --gtest_list_tests takes into account the\n    --gtest_filter flag.'
        self.RunAndVerify(flag_value='1', expected_output_re=EXPECTED_OUTPUT_FILTER_FOO_RE, other_flag='--gtest_filter=Foo*')
if __name__ == '__main__':
    gtest_test_utils.Main()