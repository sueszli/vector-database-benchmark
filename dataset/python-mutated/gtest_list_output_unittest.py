"""Unit test for Google Test's --gtest_list_tests flag.

A user can ask Google Test to list all tests by specifying the
--gtest_list_tests flag. If output is requested, via --gtest_output=xml
or --gtest_output=json, the tests are listed, with extra information in the
output file.
This script tests such functionality by invoking gtest_list_output_unittest_
 (a program written with Google Test) the command line flags.
"""
import os
import re
import gtest_test_utils
GTEST_LIST_TESTS_FLAG = '--gtest_list_tests'
GTEST_OUTPUT_FLAG = '--gtest_output'
EXPECTED_XML = '<\\?xml version="1.0" encoding="UTF-8"\\?>\n<testsuites tests="2" name="AllTests">\n  <testsuite name="FooTest" tests="2">\n    <testcase name="Test1" file=".*gtest_list_output_unittest_.cc" line="43" />\n    <testcase name="Test2" file=".*gtest_list_output_unittest_.cc" line="45" />\n  </testsuite>\n</testsuites>\n'
EXPECTED_JSON = '{\n  "tests": 2,\n  "name": "AllTests",\n  "testsuites": \\[\n    {\n      "name": "FooTest",\n      "tests": 2,\n      "testsuite": \\[\n        {\n          "name": "Test1",\n          "file": ".*gtest_list_output_unittest_.cc",\n          "line": 43\n        },\n        {\n          "name": "Test2",\n          "file": ".*gtest_list_output_unittest_.cc",\n          "line": 45\n        }\n      \\]\n    }\n  \\]\n}\n'

class GTestListTestsOutputUnitTest(gtest_test_utils.TestCase):
    """Unit test for Google Test's list tests with output to file functionality.
  """

    def testXml(self):
        if False:
            print('Hello World!')
        'Verifies XML output for listing tests in a Google Test binary.\n\n    Runs a test program that generates an empty XML output, and\n    tests that the XML output is expected.\n    '
        self._TestOutput('xml', EXPECTED_XML)

    def testJSON(self):
        if False:
            for i in range(10):
                print('nop')
        'Verifies XML output for listing tests in a Google Test binary.\n\n    Runs a test program that generates an empty XML output, and\n    tests that the XML output is expected.\n    '
        self._TestOutput('json', EXPECTED_JSON)

    def _GetOutput(self, out_format):
        if False:
            while True:
                i = 10
        file_path = os.path.join(gtest_test_utils.GetTempDir(), 'test_out.' + out_format)
        gtest_prog_path = gtest_test_utils.GetTestExecutablePath('gtest_list_output_unittest_')
        command = [gtest_prog_path, '%s=%s:%s' % (GTEST_OUTPUT_FLAG, out_format, file_path), '--gtest_list_tests']
        environ_copy = os.environ.copy()
        p = gtest_test_utils.Subprocess(command, env=environ_copy, working_dir=gtest_test_utils.GetTempDir())
        self.assert_(p.exited)
        self.assertEquals(0, p.exit_code)
        with open(file_path) as f:
            result = f.read()
        return result

    def _TestOutput(self, test_format, expected_output):
        if False:
            print('Hello World!')
        actual = self._GetOutput(test_format)
        actual_lines = actual.splitlines()
        expected_lines = expected_output.splitlines()
        line_count = 0
        for actual_line in actual_lines:
            expected_line = expected_lines[line_count]
            expected_line_re = re.compile(expected_line.strip())
            self.assert_(expected_line_re.match(actual_line.strip()), 'actual output of "%s",\nwhich does not match expected regex of "%s"\non line %d' % (actual, expected_output, line_count))
            line_count = line_count + 1
if __name__ == '__main__':
    os.environ['GTEST_STACK_TRACE_DEPTH'] = '1'
    gtest_test_utils.Main()