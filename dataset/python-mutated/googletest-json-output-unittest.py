"""Unit test for the gtest_json_output module."""
import datetime
import errno
import json
import os
import re
import sys
import gtest_json_test_utils
import gtest_test_utils
GTEST_FILTER_FLAG = '--gtest_filter'
GTEST_LIST_TESTS_FLAG = '--gtest_list_tests'
GTEST_OUTPUT_FLAG = '--gtest_output'
GTEST_DEFAULT_OUTPUT_FILE = 'test_detail.json'
GTEST_PROGRAM_NAME = 'gtest_xml_output_unittest_'
NO_STACKTRACE_SUPPORT_FLAG = '--no_stacktrace_support'
SUPPORTS_STACK_TRACES = NO_STACKTRACE_SUPPORT_FLAG not in sys.argv
if SUPPORTS_STACK_TRACES:
    STACK_TRACE_TEMPLATE = '\nStack trace:\n*'
else:
    STACK_TRACE_TEMPLATE = ''
EXPECTED_NON_EMPTY = {u'tests': 24, u'failures': 4, u'disabled': 2, u'errors': 0, u'timestamp': u'*', u'time': u'*', u'ad_hoc_property': u'42', u'name': u'AllTests', u'testsuites': [{u'name': u'SuccessfulTest', u'tests': 1, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'Succeeds', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'SuccessfulTest'}]}, {u'name': u'FailedTest', u'tests': 1, u'failures': 1, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'Fails', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'FailedTest', u'failures': [{u'failure': u'gtest_xml_output_unittest_.cc:*\nExpected equality of these values:\n  1\n  2' + STACK_TRACE_TEMPLATE, u'type': u''}]}]}, {u'name': u'DisabledTest', u'tests': 1, u'failures': 0, u'disabled': 1, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'DISABLED_test_not_run', u'status': u'NOTRUN', u'result': u'SUPPRESSED', u'time': u'*', u'classname': u'DisabledTest'}]}, {u'name': u'SkippedTest', u'tests': 1, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'Skipped', u'status': u'RUN', u'result': u'SKIPPED', u'time': u'*', u'classname': u'SkippedTest'}]}, {u'name': u'MixedResultTest', u'tests': 3, u'failures': 1, u'disabled': 1, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'Succeeds', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'MixedResultTest'}, {u'name': u'Fails', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'MixedResultTest', u'failures': [{u'failure': u'gtest_xml_output_unittest_.cc:*\nExpected equality of these values:\n  1\n  2' + STACK_TRACE_TEMPLATE, u'type': u''}, {u'failure': u'gtest_xml_output_unittest_.cc:*\nExpected equality of these values:\n  2\n  3' + STACK_TRACE_TEMPLATE, u'type': u''}]}, {u'name': u'DISABLED_test', u'status': u'NOTRUN', u'result': u'SUPPRESSED', u'time': u'*', u'classname': u'MixedResultTest'}]}, {u'name': u'XmlQuotingTest', u'tests': 1, u'failures': 1, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'OutputsCData', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'XmlQuotingTest', u'failures': [{u'failure': u'gtest_xml_output_unittest_.cc:*\nFailed\nXML output: <?xml encoding="utf-8"><top><![CDATA[cdata text]]></top>' + STACK_TRACE_TEMPLATE, u'type': u''}]}]}, {u'name': u'InvalidCharactersTest', u'tests': 1, u'failures': 1, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'InvalidCharactersInMessage', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'InvalidCharactersTest', u'failures': [{u'failure': u'gtest_xml_output_unittest_.cc:*\nFailed\nInvalid characters in brackets [\x01\x02]' + STACK_TRACE_TEMPLATE, u'type': u''}]}]}, {u'name': u'PropertyRecordingTest', u'tests': 4, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'SetUpTestSuite': u'yes', u'TearDownTestSuite': u'aye', u'testsuite': [{u'name': u'OneProperty', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'PropertyRecordingTest', u'key_1': u'1'}, {u'name': u'IntValuedProperty', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'PropertyRecordingTest', u'key_int': u'1'}, {u'name': u'ThreeProperties', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'PropertyRecordingTest', u'key_1': u'1', u'key_2': u'2', u'key_3': u'3'}, {u'name': u'TwoValuesForOneKeyUsesLastValue', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'PropertyRecordingTest', u'key_1': u'2'}]}, {u'name': u'NoFixtureTest', u'tests': 3, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'RecordProperty', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'NoFixtureTest', u'key': u'1'}, {u'name': u'ExternalUtilityThatCallsRecordIntValuedProperty', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'NoFixtureTest', u'key_for_utility_int': u'1'}, {u'name': u'ExternalUtilityThatCallsRecordStringValuedProperty', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'NoFixtureTest', u'key_for_utility_string': u'1'}]}, {u'name': u'TypedTest/0', u'tests': 1, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'HasTypeParamAttribute', u'type_param': u'int', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'TypedTest/0'}]}, {u'name': u'TypedTest/1', u'tests': 1, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'HasTypeParamAttribute', u'type_param': u'long', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'TypedTest/1'}]}, {u'name': u'Single/TypeParameterizedTestSuite/0', u'tests': 1, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'HasTypeParamAttribute', u'type_param': u'int', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'Single/TypeParameterizedTestSuite/0'}]}, {u'name': u'Single/TypeParameterizedTestSuite/1', u'tests': 1, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'HasTypeParamAttribute', u'type_param': u'long', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'Single/TypeParameterizedTestSuite/1'}]}, {u'name': u'Single/ValueParamTest', u'tests': 4, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'HasValueParamAttribute/0', u'value_param': u'33', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'Single/ValueParamTest'}, {u'name': u'HasValueParamAttribute/1', u'value_param': u'42', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'Single/ValueParamTest'}, {u'name': u'AnotherTestThatHasValueParamAttribute/0', u'value_param': u'33', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'Single/ValueParamTest'}, {u'name': u'AnotherTestThatHasValueParamAttribute/1', u'value_param': u'42', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'Single/ValueParamTest'}]}]}
EXPECTED_FILTERED = {u'tests': 1, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'timestamp': u'*', u'name': u'AllTests', u'ad_hoc_property': u'42', u'testsuites': [{u'name': u'SuccessfulTest', u'tests': 1, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'Succeeds', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'SuccessfulTest'}]}]}
EXPECTED_EMPTY = {u'tests': 0, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'timestamp': u'*', u'name': u'AllTests', u'testsuites': []}
GTEST_PROGRAM_PATH = gtest_test_utils.GetTestExecutablePath(GTEST_PROGRAM_NAME)
SUPPORTS_TYPED_TESTS = 'TypedTest' in gtest_test_utils.Subprocess([GTEST_PROGRAM_PATH, GTEST_LIST_TESTS_FLAG], capture_stderr=False).output

class GTestJsonOutputUnitTest(gtest_test_utils.TestCase):
    """Unit test for Google Test's JSON output functionality.
  """
    if SUPPORTS_TYPED_TESTS:

        def testNonEmptyJsonOutput(self):
            if False:
                for i in range(10):
                    print('nop')
            'Verifies JSON output for a Google Test binary with non-empty output.\n\n      Runs a test program that generates a non-empty JSON output, and\n      tests that the JSON output is expected.\n      '
            self._TestJsonOutput(GTEST_PROGRAM_NAME, EXPECTED_NON_EMPTY, 1)

    def testEmptyJsonOutput(self):
        if False:
            while True:
                i = 10
        'Verifies JSON output for a Google Test binary without actual tests.\n\n    Runs a test program that generates an empty JSON output, and\n    tests that the JSON output is expected.\n    '
        self._TestJsonOutput('gtest_no_test_unittest', EXPECTED_EMPTY, 0)

    def testTimestampValue(self):
        if False:
            return 10
        'Checks whether the timestamp attribute in the JSON output is valid.\n\n    Runs a test program that generates an empty JSON output, and checks if\n    the timestamp attribute in the testsuites tag is valid.\n    '
        actual = self._GetJsonOutput('gtest_no_test_unittest', [], 0)
        date_time_str = actual['timestamp']
        match = re.match('(\\d+)-(\\d\\d)-(\\d\\d)T(\\d\\d):(\\d\\d):(\\d\\d)', date_time_str)
        self.assertTrue(re.match, 'JSON datettime string %s has incorrect format' % date_time_str)
        date_time_from_json = datetime.datetime(year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3)), hour=int(match.group(4)), minute=int(match.group(5)), second=int(match.group(6)))
        time_delta = abs(datetime.datetime.now() - date_time_from_json)
        self.assertTrue(time_delta < datetime.timedelta(seconds=600), 'time_delta is %s' % time_delta)

    def testDefaultOutputFile(self):
        if False:
            i = 10
            return i + 15
        'Verifies the default output file name.\n\n    Confirms that Google Test produces an JSON output file with the expected\n    default name if no name is explicitly specified.\n    '
        output_file = os.path.join(gtest_test_utils.GetTempDir(), GTEST_DEFAULT_OUTPUT_FILE)
        gtest_prog_path = gtest_test_utils.GetTestExecutablePath('gtest_no_test_unittest')
        try:
            os.remove(output_file)
        except OSError:
            e = sys.exc_info()[1]
            if e.errno != errno.ENOENT:
                raise
        p = gtest_test_utils.Subprocess([gtest_prog_path, '%s=json' % GTEST_OUTPUT_FLAG], working_dir=gtest_test_utils.GetTempDir())
        self.assert_(p.exited)
        self.assertEquals(0, p.exit_code)
        self.assert_(os.path.isfile(output_file))

    def testSuppressedJsonOutput(self):
        if False:
            return 10
        'Verifies that no JSON output is generated.\n\n    Tests that no JSON file is generated if the default JSON listener is\n    shut down before RUN_ALL_TESTS is invoked.\n    '
        json_path = os.path.join(gtest_test_utils.GetTempDir(), GTEST_PROGRAM_NAME + 'out.json')
        if os.path.isfile(json_path):
            os.remove(json_path)
        command = [GTEST_PROGRAM_PATH, '%s=json:%s' % (GTEST_OUTPUT_FLAG, json_path), '--shut_down_xml']
        p = gtest_test_utils.Subprocess(command)
        if p.terminated_by_signal:
            self.assertFalse(p.terminated_by_signal, '%s was killed by signal %d' % (GTEST_PROGRAM_NAME, p.signal))
        else:
            self.assert_(p.exited)
            self.assertEquals(1, p.exit_code, "'%s' exited with code %s, which doesn't match the expected exit code %s." % (command, p.exit_code, 1))
        self.assert_(not os.path.isfile(json_path))

    def testFilteredTestJsonOutput(self):
        if False:
            print('Hello World!')
        'Verifies JSON output when a filter is applied.\n\n    Runs a test program that executes only some tests and verifies that\n    non-selected tests do not show up in the JSON output.\n    '
        self._TestJsonOutput(GTEST_PROGRAM_NAME, EXPECTED_FILTERED, 0, extra_args=['%s=SuccessfulTest.*' % GTEST_FILTER_FLAG])

    def _GetJsonOutput(self, gtest_prog_name, extra_args, expected_exit_code):
        if False:
            for i in range(10):
                print('nop')
        "Returns the JSON output generated by running the program gtest_prog_name.\n\n    Furthermore, the program's exit code must be expected_exit_code.\n\n    Args:\n      gtest_prog_name: Google Test binary name.\n      extra_args: extra arguments to binary invocation.\n      expected_exit_code: program's exit code.\n    "
        json_path = os.path.join(gtest_test_utils.GetTempDir(), gtest_prog_name + 'out.json')
        gtest_prog_path = gtest_test_utils.GetTestExecutablePath(gtest_prog_name)
        command = [gtest_prog_path, '%s=json:%s' % (GTEST_OUTPUT_FLAG, json_path)] + extra_args
        p = gtest_test_utils.Subprocess(command)
        if p.terminated_by_signal:
            self.assert_(False, '%s was killed by signal %d' % (gtest_prog_name, p.signal))
        else:
            self.assert_(p.exited)
            self.assertEquals(expected_exit_code, p.exit_code, "'%s' exited with code %s, which doesn't match the expected exit code %s." % (command, p.exit_code, expected_exit_code))
        with open(json_path) as f:
            actual = json.load(f)
        return actual

    def _TestJsonOutput(self, gtest_prog_name, expected, expected_exit_code, extra_args=None):
        if False:
            print('Hello World!')
        "Checks the JSON output generated by the Google Test binary.\n\n    Asserts that the JSON document generated by running the program\n    gtest_prog_name matches expected_json, a string containing another\n    JSON document.  Furthermore, the program's exit code must be\n    expected_exit_code.\n\n    Args:\n      gtest_prog_name: Google Test binary name.\n      expected: expected output.\n      expected_exit_code: program's exit code.\n      extra_args: extra arguments to binary invocation.\n    "
        actual = self._GetJsonOutput(gtest_prog_name, extra_args or [], expected_exit_code)
        self.assertEqual(expected, gtest_json_test_utils.normalize(actual))
if __name__ == '__main__':
    if NO_STACKTRACE_SUPPORT_FLAG in sys.argv:
        sys.argv.remove(NO_STACKTRACE_SUPPORT_FLAG)
    os.environ['GTEST_STACK_TRACE_DEPTH'] = '1'
    gtest_test_utils.Main()