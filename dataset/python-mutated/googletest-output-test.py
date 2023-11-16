"""Tests the text output of Google C++ Testing and Mocking Framework.

To update the golden file:
googletest_output_test.py --build_dir=BUILD/DIR --gengolden
where BUILD/DIR contains the built googletest-output-test_ file.
googletest_output_test.py --gengolden
googletest_output_test.py
"""
import difflib
import os
import re
import sys
import gtest_test_utils
GENGOLDEN_FLAG = '--gengolden'
CATCH_EXCEPTIONS_ENV_VAR_NAME = 'GTEST_CATCH_EXCEPTIONS'
NO_STACKTRACE_SUPPORT_FLAG = '--no_stacktrace_support'
IS_LINUX = os.name == 'posix' and os.uname()[0] == 'Linux'
IS_WINDOWS = os.name == 'nt'
GOLDEN_NAME = 'googletest-output-test-golden-lin.txt'
PROGRAM_PATH = gtest_test_utils.GetTestExecutablePath('googletest-output-test_')
COMMAND_LIST_TESTS = ({}, [PROGRAM_PATH, '--gtest_list_tests'])
COMMAND_WITH_COLOR = ({}, [PROGRAM_PATH, '--gtest_color=yes'])
COMMAND_WITH_TIME = ({}, [PROGRAM_PATH, '--gtest_print_time', 'internal_skip_environment_and_ad_hoc_tests', '--gtest_filter=FatalFailureTest.*:LoggingTest.*'])
COMMAND_WITH_DISABLED = ({}, [PROGRAM_PATH, '--gtest_also_run_disabled_tests', 'internal_skip_environment_and_ad_hoc_tests', '--gtest_filter=*DISABLED_*'])
COMMAND_WITH_SHARDING = ({'GTEST_SHARD_INDEX': '1', 'GTEST_TOTAL_SHARDS': '2'}, [PROGRAM_PATH, 'internal_skip_environment_and_ad_hoc_tests', '--gtest_filter=PassingTest.*'])
GOLDEN_PATH = os.path.join(gtest_test_utils.GetSourceDir(), GOLDEN_NAME)

def ToUnixLineEnding(s):
    if False:
        print('Hello World!')
    'Changes all Windows/Mac line endings in s to UNIX line endings.'
    return s.replace('\r\n', '\n').replace('\r', '\n')

def RemoveLocations(test_output):
    if False:
        for i in range(10):
            print('nop')
    "Removes all file location info from a Google Test program's output.\n\n  Args:\n       test_output:  the output of a Google Test program.\n\n  Returns:\n       output with all file location info (in the form of\n       'DIRECTORY/FILE_NAME:LINE_NUMBER: 'or\n       'DIRECTORY\\FILE_NAME(LINE_NUMBER): ') replaced by\n       'FILE_NAME:#: '.\n  "
    return re.sub('.*[/\\\\]((googletest-output-test_|gtest).cc)(\\:\\d+|\\(\\d+\\))\\: ', '\\1:#: ', test_output)

def RemoveStackTraceDetails(output):
    if False:
        return 10
    "Removes all stack traces from a Google Test program's output."
    return re.sub('Stack trace:(.|\\n)*?\\n\\n', 'Stack trace: (omitted)\n\n', output)

def RemoveStackTraces(output):
    if False:
        while True:
            i = 10
    "Removes all traces of stack traces from a Google Test program's output."
    return re.sub('Stack trace:(.|\\n)*?\\n\\n', '', output)

def RemoveTime(output):
    if False:
        return 10
    "Removes all time information from a Google Test program's output."
    return re.sub('\\(\\d+ ms', '(? ms', output)

def RemoveTypeInfoDetails(test_output):
    if False:
        return 10
    "Removes compiler-specific type info from Google Test program's output.\n\n  Args:\n       test_output:  the output of a Google Test program.\n\n  Returns:\n       output with type information normalized to canonical form.\n  "
    return re.sub('unsigned int', 'unsigned', test_output)

def NormalizeToCurrentPlatform(test_output):
    if False:
        for i in range(10):
            print('nop')
    'Normalizes platform specific output details for easier comparison.'
    if IS_WINDOWS:
        test_output = re.sub('\x1b\\[(0;3\\d)?m', '', test_output)
        test_output = re.sub(': Failure\\n', ': error: ', test_output)
        test_output = re.sub('((\\w|\\.)+)\\((\\d+)\\):', '\\1:\\3:', test_output)
    return test_output

def RemoveTestCounts(output):
    if False:
        while True:
            i = 10
    "Removes test counts from a Google Test program's output."
    output = re.sub('\\d+ tests?, listed below', '? tests, listed below', output)
    output = re.sub('\\d+ FAILED TESTS', '? FAILED TESTS', output)
    output = re.sub('\\d+ tests? from \\d+ test cases?', '? tests from ? test cases', output)
    output = re.sub('\\d+ tests? from ([a-zA-Z_])', '? tests from \\1', output)
    return re.sub('\\d+ tests?\\.', '? tests.', output)

def RemoveMatchingTests(test_output, pattern):
    if False:
        print('Hello World!')
    "Removes output of specified tests from a Google Test program's output.\n\n  This function strips not only the beginning and the end of a test but also\n  all output in between.\n\n  Args:\n    test_output:       A string containing the test output.\n    pattern:           A regex string that matches names of test cases or\n                       tests to remove.\n\n  Returns:\n    Contents of test_output with tests whose names match pattern removed.\n  "
    test_output = re.sub('.*\\[ RUN      \\] .*%s(.|\\n)*?\\[(  FAILED  |       OK )\\] .*%s.*\\n' % (pattern, pattern), '', test_output)
    return re.sub('.*%s.*\\n' % pattern, '', test_output)

def NormalizeOutput(output):
    if False:
        return 10
    'Normalizes output (the output of googletest-output-test_.exe).'
    output = ToUnixLineEnding(output)
    output = RemoveLocations(output)
    output = RemoveStackTraceDetails(output)
    output = RemoveTime(output)
    return output

def GetShellCommandOutput(env_cmd):
    if False:
        while True:
            i = 10
    "Runs a command in a sub-process, and returns its output in a string.\n\n  Args:\n    env_cmd: The shell command. A 2-tuple where element 0 is a dict of extra\n             environment variables to set, and element 1 is a string with\n             the command and any flags.\n\n  Returns:\n    A string with the command's combined standard and diagnostic output.\n  "
    environ = os.environ.copy()
    environ.update(env_cmd[0])
    p = gtest_test_utils.Subprocess(env_cmd[1], env=environ)
    return p.output

def GetCommandOutput(env_cmd):
    if False:
        print('Hello World!')
    'Runs a command and returns its output with all file location\n  info stripped off.\n\n  Args:\n    env_cmd:  The shell command. A 2-tuple where element 0 is a dict of extra\n              environment variables to set, and element 1 is a string with\n              the command and any flags.\n  '
    (environ, cmdline) = env_cmd
    environ = dict(environ)
    environ[CATCH_EXCEPTIONS_ENV_VAR_NAME] = '1'
    return NormalizeOutput(GetShellCommandOutput((environ, cmdline)))

def GetOutputOfAllCommands():
    if False:
        return 10
    'Returns concatenated output from several representative commands.'
    return GetCommandOutput(COMMAND_WITH_COLOR) + GetCommandOutput(COMMAND_WITH_TIME) + GetCommandOutput(COMMAND_WITH_DISABLED) + GetCommandOutput(COMMAND_WITH_SHARDING)
test_list = GetShellCommandOutput(COMMAND_LIST_TESTS)
SUPPORTS_DEATH_TESTS = 'DeathTest' in test_list
SUPPORTS_TYPED_TESTS = 'TypedTest' in test_list
SUPPORTS_THREADS = 'ExpectFailureWithThreadsTest' in test_list
SUPPORTS_STACK_TRACES = NO_STACKTRACE_SUPPORT_FLAG not in sys.argv
CAN_GENERATE_GOLDEN_FILE = SUPPORTS_DEATH_TESTS and SUPPORTS_TYPED_TESTS and SUPPORTS_THREADS and SUPPORTS_STACK_TRACES

class GTestOutputTest(gtest_test_utils.TestCase):

    def RemoveUnsupportedTests(self, test_output):
        if False:
            print('Hello World!')
        if not SUPPORTS_DEATH_TESTS:
            test_output = RemoveMatchingTests(test_output, 'DeathTest')
        if not SUPPORTS_TYPED_TESTS:
            test_output = RemoveMatchingTests(test_output, 'TypedTest')
            test_output = RemoveMatchingTests(test_output, 'TypedDeathTest')
            test_output = RemoveMatchingTests(test_output, 'TypeParamDeathTest')
        if not SUPPORTS_THREADS:
            test_output = RemoveMatchingTests(test_output, 'ExpectFailureWithThreadsTest')
            test_output = RemoveMatchingTests(test_output, 'ScopedFakeTestPartResultReporterTest')
            test_output = RemoveMatchingTests(test_output, 'WorksConcurrently')
        if not SUPPORTS_STACK_TRACES:
            test_output = RemoveStackTraces(test_output)
        return test_output

    def testOutput(self):
        if False:
            for i in range(10):
                print('nop')
        output = GetOutputOfAllCommands()
        golden_file = open(GOLDEN_PATH, 'rb')
        golden = ToUnixLineEnding(golden_file.read().decode())
        golden_file.close()
        normalized_actual = RemoveTypeInfoDetails(output)
        normalized_golden = RemoveTypeInfoDetails(golden)
        if CAN_GENERATE_GOLDEN_FILE:
            self.assertEqual(normalized_golden, normalized_actual, '\n'.join(difflib.unified_diff(normalized_golden.split('\n'), normalized_actual.split('\n'), 'golden', 'actual')))
        else:
            normalized_actual = NormalizeToCurrentPlatform(RemoveTestCounts(normalized_actual))
            normalized_golden = NormalizeToCurrentPlatform(RemoveTestCounts(self.RemoveUnsupportedTests(normalized_golden)))
            if os.getenv('DEBUG_GTEST_OUTPUT_TEST'):
                open(os.path.join(gtest_test_utils.GetSourceDir(), '_googletest-output-test_normalized_actual.txt'), 'wb').write(normalized_actual)
                open(os.path.join(gtest_test_utils.GetSourceDir(), '_googletest-output-test_normalized_golden.txt'), 'wb').write(normalized_golden)
            self.assertEqual(normalized_golden, normalized_actual)
if __name__ == '__main__':
    if NO_STACKTRACE_SUPPORT_FLAG in sys.argv:
        sys.argv.remove(NO_STACKTRACE_SUPPORT_FLAG)
    if GENGOLDEN_FLAG in sys.argv:
        if CAN_GENERATE_GOLDEN_FILE:
            output = GetOutputOfAllCommands()
            golden_file = open(GOLDEN_PATH, 'wb')
            golden_file.write(output)
            golden_file.close()
        else:
            message = 'Unable to write a golden file when compiled in an environment\nthat does not support all the required features (death tests,\ntyped tests, stack traces, and multiple threads).\nPlease build this test and generate the golden file using Blaze on Linux.'
            sys.stderr.write(message)
            sys.exit(1)
    else:
        gtest_test_utils.Main()