"""Unit test utilities for Google C++ Mocking Framework."""
import os
import sys
SCRIPT_DIR = os.path.dirname(__file__) or '.'
gtest_tests_util_dir = os.path.join(SCRIPT_DIR, '../../googletest/test')
if os.path.isdir(gtest_tests_util_dir):
    GTEST_TESTS_UTIL_DIR = gtest_tests_util_dir
else:
    GTEST_TESTS_UTIL_DIR = os.path.join(SCRIPT_DIR, '../../googletest/test')
sys.path.append(GTEST_TESTS_UTIL_DIR)
import gtest_test_utils

def GetSourceDir():
    if False:
        while True:
            i = 10
    'Returns the absolute path of the directory where the .py files are.'
    return gtest_test_utils.GetSourceDir()

def GetTestExecutablePath(executable_name):
    if False:
        return 10
    "Returns the absolute path of the test binary given its name.\n\n  The function will print a message and abort the program if the resulting file\n  doesn't exist.\n\n  Args:\n    executable_name: name of the test binary that the test script runs.\n\n  Returns:\n    The absolute path of the test binary.\n  "
    return gtest_test_utils.GetTestExecutablePath(executable_name)

def GetExitStatus(exit_code):
    if False:
        i = 10
        return i + 15
    "Returns the argument to exit(), or -1 if exit() wasn't called.\n\n  Args:\n    exit_code: the result value of os.system(command).\n  "
    if os.name == 'nt':
        return exit_code
    elif os.WIFEXITED(exit_code):
        return os.WEXITSTATUS(exit_code)
    else:
        return -1
Subprocess = gtest_test_utils.Subprocess
TestCase = gtest_test_utils.TestCase
environ = gtest_test_utils.environ
SetEnvVar = gtest_test_utils.SetEnvVar
PREMATURE_EXIT_FILE_ENV_VAR = gtest_test_utils.PREMATURE_EXIT_FILE_ENV_VAR

def Main():
    if False:
        for i in range(10):
            print('nop')
    'Runs the unit test.'
    gtest_test_utils.Main()