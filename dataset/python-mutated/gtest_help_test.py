"""Tests the --help flag of Google C++ Testing and Mocking Framework.

SYNOPSIS
       gtest_help_test.py --build_dir=BUILD/DIR
         # where BUILD/DIR contains the built gtest_help_test_ file.
       gtest_help_test.py
"""
import os
import re
import gtest_test_utils
IS_LINUX = os.name == 'posix' and os.uname()[0] == 'Linux'
IS_WINDOWS = os.name == 'nt'
PROGRAM_PATH = gtest_test_utils.GetTestExecutablePath('gtest_help_test_')
FLAG_PREFIX = '--gtest_'
DEATH_TEST_STYLE_FLAG = FLAG_PREFIX + 'death_test_style'
STREAM_RESULT_TO_FLAG = FLAG_PREFIX + 'stream_result_to'
UNKNOWN_FLAG = FLAG_PREFIX + 'unknown_flag_for_testing'
LIST_TESTS_FLAG = FLAG_PREFIX + 'list_tests'
INCORRECT_FLAG_VARIANTS = [re.sub('^--', '-', LIST_TESTS_FLAG), re.sub('^--', '/', LIST_TESTS_FLAG), re.sub('_', '-', LIST_TESTS_FLAG)]
INTERNAL_FLAG_FOR_TESTING = FLAG_PREFIX + 'internal_flag_for_testing'
SUPPORTS_DEATH_TESTS = 'DeathTest' in gtest_test_utils.Subprocess([PROGRAM_PATH, LIST_TESTS_FLAG]).output
HELP_REGEX = re.compile(FLAG_PREFIX + 'list_tests.*' + FLAG_PREFIX + 'filter=.*' + FLAG_PREFIX + 'also_run_disabled_tests.*' + FLAG_PREFIX + 'repeat=.*' + FLAG_PREFIX + 'shuffle.*' + FLAG_PREFIX + 'random_seed=.*' + FLAG_PREFIX + 'color=.*' + FLAG_PREFIX + 'print_time.*' + FLAG_PREFIX + 'output=.*' + FLAG_PREFIX + 'break_on_failure.*' + FLAG_PREFIX + 'throw_on_failure.*' + FLAG_PREFIX + 'catch_exceptions=0.*', re.DOTALL)

def RunWithFlag(flag):
    if False:
        while True:
            i = 10
    'Runs gtest_help_test_ with the given flag.\n\n  Returns:\n    the exit code and the text output as a tuple.\n  Args:\n    flag: the command-line flag to pass to gtest_help_test_, or None.\n  '
    if flag is None:
        command = [PROGRAM_PATH]
    else:
        command = [PROGRAM_PATH, flag]
    child = gtest_test_utils.Subprocess(command)
    return (child.exit_code, child.output)

class GTestHelpTest(gtest_test_utils.TestCase):
    """Tests the --help flag and its equivalent forms."""

    def TestHelpFlag(self, flag):
        if False:
            for i in range(10):
                print('nop')
        'Verifies correct behavior when help flag is specified.\n\n    The right message must be printed and the tests must\n    skipped when the given flag is specified.\n\n    Args:\n      flag:  A flag to pass to the binary or None.\n    '
        (exit_code, output) = RunWithFlag(flag)
        self.assertEquals(0, exit_code)
        self.assert_(HELP_REGEX.search(output), output)
        if IS_LINUX:
            self.assert_(STREAM_RESULT_TO_FLAG in output, output)
        else:
            self.assert_(STREAM_RESULT_TO_FLAG not in output, output)
        if SUPPORTS_DEATH_TESTS and (not IS_WINDOWS):
            self.assert_(DEATH_TEST_STYLE_FLAG in output, output)
        else:
            self.assert_(DEATH_TEST_STYLE_FLAG not in output, output)

    def TestNonHelpFlag(self, flag):
        if False:
            i = 10
            return i + 15
        'Verifies correct behavior when no help flag is specified.\n\n    Verifies that when no help flag is specified, the tests are run\n    and the help message is not printed.\n\n    Args:\n      flag:  A flag to pass to the binary or None.\n    '
        (exit_code, output) = RunWithFlag(flag)
        self.assert_(exit_code != 0)
        self.assert_(not HELP_REGEX.search(output), output)

    def testPrintsHelpWithFullFlag(self):
        if False:
            while True:
                i = 10
        self.TestHelpFlag('--help')

    def testPrintsHelpWithShortFlag(self):
        if False:
            return 10
        self.TestHelpFlag('-h')

    def testPrintsHelpWithQuestionFlag(self):
        if False:
            return 10
        self.TestHelpFlag('-?')

    def testPrintsHelpWithWindowsStyleQuestionFlag(self):
        if False:
            print('Hello World!')
        self.TestHelpFlag('/?')

    def testPrintsHelpWithUnrecognizedGoogleTestFlag(self):
        if False:
            print('Hello World!')
        self.TestHelpFlag(UNKNOWN_FLAG)

    def testPrintsHelpWithIncorrectFlagStyle(self):
        if False:
            for i in range(10):
                print('nop')
        for incorrect_flag in INCORRECT_FLAG_VARIANTS:
            self.TestHelpFlag(incorrect_flag)

    def testRunsTestsWithoutHelpFlag(self):
        if False:
            print('Hello World!')
        'Verifies that when no help flag is specified, the tests are run\n    and the help message is not printed.'
        self.TestNonHelpFlag(None)

    def testRunsTestsWithGtestInternalFlag(self):
        if False:
            print('Hello World!')
        "Verifies that the tests are run and no help message is printed when\n    a flag starting with Google Test prefix and 'internal_' is supplied."
        self.TestNonHelpFlag(INTERNAL_FLAG_FOR_TESTING)
if __name__ == '__main__':
    gtest_test_utils.Main()