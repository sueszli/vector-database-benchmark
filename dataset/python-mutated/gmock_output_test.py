"""Tests the text output of Google C++ Mocking Framework.

To update the golden file:
gmock_output_test.py --build_dir=BUILD/DIR --gengolden
where BUILD/DIR contains the built gmock_output_test_ file.
gmock_output_test.py --gengolden
gmock_output_test.py

"""
import os
import re
import sys
import gmock_test_utils
GENGOLDEN_FLAG = '--gengolden'
PROGRAM_PATH = gmock_test_utils.GetTestExecutablePath('gmock_output_test_')
COMMAND = [PROGRAM_PATH, '--gtest_stack_trace_depth=0', '--gtest_print_time=0']
GOLDEN_NAME = 'gmock_output_test_golden.txt'
GOLDEN_PATH = os.path.join(gmock_test_utils.GetSourceDir(), GOLDEN_NAME)

def ToUnixLineEnding(s):
    if False:
        for i in range(10):
            print('nop')
    'Changes all Windows/Mac line endings in s to UNIX line endings.'
    return s.replace('\r\n', '\n').replace('\r', '\n')

def RemoveReportHeaderAndFooter(output):
    if False:
        return 10
    "Removes Google Test result report's header and footer from the output."
    output = re.sub('.*gtest_main.*\\n', '', output)
    output = re.sub('\\[.*\\d+ tests.*\\n', '', output)
    output = re.sub('\\[.* test environment .*\\n', '', output)
    output = re.sub('\\[=+\\] \\d+ tests .* ran.*', '', output)
    output = re.sub('.* FAILED TESTS\\n', '', output)
    return output

def RemoveLocations(output):
    if False:
        return 10
    "Removes all file location info from a Google Test program's output.\n\n  Args:\n       output:  the output of a Google Test program.\n\n  Returns:\n       output with all file location info (in the form of\n       'DIRECTORY/FILE_NAME:LINE_NUMBER: 'or\n       'DIRECTORY\\FILE_NAME(LINE_NUMBER): ') replaced by\n       'FILE:#: '.\n  "
    return re.sub('.*[/\\\\](.+)(\\:\\d+|\\(\\d+\\))\\:', 'FILE:#:', output)

def NormalizeErrorMarker(output):
    if False:
        while True:
            i = 10
    'Normalizes the error marker, which is different on Windows vs on Linux.'
    return re.sub(' error: ', ' Failure\n', output)

def RemoveMemoryAddresses(output):
    if False:
        i = 10
        return i + 15
    'Removes memory addresses from the test output.'
    return re.sub('@\\w+', '@0x#', output)

def RemoveTestNamesOfLeakedMocks(output):
    if False:
        i = 10
        return i + 15
    'Removes the test names of leaked mock objects from the test output.'
    return re.sub('\\(used in test .+\\) ', '', output)

def GetLeakyTests(output):
    if False:
        while True:
            i = 10
    'Returns a list of test names that leak mock objects.'
    return re.findall('\\(used in test (.+)\\)', output)

def GetNormalizedOutputAndLeakyTests(output):
    if False:
        for i in range(10):
            print('nop')
    'Normalizes the output of gmock_output_test_.\n\n  Args:\n    output: The test output.\n\n  Returns:\n    A tuple (the normalized test output, the list of test names that have\n    leaked mocks).\n  '
    output = ToUnixLineEnding(output)
    output = RemoveReportHeaderAndFooter(output)
    output = NormalizeErrorMarker(output)
    output = RemoveLocations(output)
    output = RemoveMemoryAddresses(output)
    return (RemoveTestNamesOfLeakedMocks(output), GetLeakyTests(output))

def GetShellCommandOutput(cmd):
    if False:
        for i in range(10):
            print('nop')
    'Runs a command in a sub-process, and returns its STDOUT in a string.'
    return gmock_test_utils.Subprocess(cmd, capture_stderr=False).output

def GetNormalizedCommandOutputAndLeakyTests(cmd):
    if False:
        return 10
    'Runs a command and returns its normalized output and a list of leaky tests.\n\n  Args:\n    cmd:  the shell command.\n  '
    os.environ['GTEST_CATCH_EXCEPTIONS'] = '1'
    return GetNormalizedOutputAndLeakyTests(GetShellCommandOutput(cmd))

class GMockOutputTest(gmock_test_utils.TestCase):

    def testOutput(self):
        if False:
            while True:
                i = 10
        (output, leaky_tests) = GetNormalizedCommandOutputAndLeakyTests(COMMAND)
        golden_file = open(GOLDEN_PATH, 'rb')
        golden = golden_file.read()
        golden_file.close()
        self.assertEquals(golden, output)
        self.assertEquals(['GMockOutputTest.CatchesLeakedMocks', 'GMockOutputTest.CatchesLeakedMocks'], leaky_tests)
if __name__ == '__main__':
    if sys.argv[1:] == [GENGOLDEN_FLAG]:
        (output, _) = GetNormalizedCommandOutputAndLeakyTests(COMMAND)
        golden_file = open(GOLDEN_PATH, 'wb')
        golden_file.write(output)
        golden_file.close()
        os._exit(0)
    else:
        gmock_test_utils.Main()