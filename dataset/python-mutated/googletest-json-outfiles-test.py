"""Unit test for the gtest_json_output module."""
import json
import os
import gtest_json_test_utils
import gtest_test_utils
GTEST_OUTPUT_SUBDIR = 'json_outfiles'
GTEST_OUTPUT_1_TEST = 'gtest_xml_outfile1_test_'
GTEST_OUTPUT_2_TEST = 'gtest_xml_outfile2_test_'
EXPECTED_1 = {u'tests': 1, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'timestamp': u'*', u'name': u'AllTests', u'testsuites': [{u'name': u'PropertyOne', u'tests': 1, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'TestSomeProperties', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'PropertyOne', u'SetUpProp': u'1', u'TestSomeProperty': u'1', u'TearDownProp': u'1'}]}]}
EXPECTED_2 = {u'tests': 1, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'timestamp': u'*', u'name': u'AllTests', u'testsuites': [{u'name': u'PropertyTwo', u'tests': 1, u'failures': 0, u'disabled': 0, u'errors': 0, u'time': u'*', u'testsuite': [{u'name': u'TestSomeProperties', u'status': u'RUN', u'result': u'COMPLETED', u'time': u'*', u'classname': u'PropertyTwo', u'SetUpProp': u'2', u'TestSomeProperty': u'2', u'TearDownProp': u'2'}]}]}

class GTestJsonOutFilesTest(gtest_test_utils.TestCase):
    """Unit test for Google Test's JSON output functionality."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.output_dir_ = os.path.join(gtest_test_utils.GetTempDir(), GTEST_OUTPUT_SUBDIR, '')
        self.DeleteFilesAndDir()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.DeleteFilesAndDir()

    def DeleteFilesAndDir(self):
        if False:
            while True:
                i = 10
        try:
            os.remove(os.path.join(self.output_dir_, GTEST_OUTPUT_1_TEST + '.json'))
        except os.error:
            pass
        try:
            os.remove(os.path.join(self.output_dir_, GTEST_OUTPUT_2_TEST + '.json'))
        except os.error:
            pass
        try:
            os.rmdir(self.output_dir_)
        except os.error:
            pass

    def testOutfile1(self):
        if False:
            return 10
        self._TestOutFile(GTEST_OUTPUT_1_TEST, EXPECTED_1)

    def testOutfile2(self):
        if False:
            for i in range(10):
                print('nop')
        self._TestOutFile(GTEST_OUTPUT_2_TEST, EXPECTED_2)

    def _TestOutFile(self, test_name, expected):
        if False:
            return 10
        gtest_prog_path = gtest_test_utils.GetTestExecutablePath(test_name)
        command = [gtest_prog_path, '--gtest_output=json:%s' % self.output_dir_]
        p = gtest_test_utils.Subprocess(command, working_dir=gtest_test_utils.GetTempDir())
        self.assert_(p.exited)
        self.assertEquals(0, p.exit_code)
        output_file_name1 = test_name + '.json'
        output_file1 = os.path.join(self.output_dir_, output_file_name1)
        output_file_name2 = 'lt-' + output_file_name1
        output_file2 = os.path.join(self.output_dir_, output_file_name2)
        self.assert_(os.path.isfile(output_file1) or os.path.isfile(output_file2), output_file1)
        if os.path.isfile(output_file1):
            with open(output_file1) as f:
                actual = json.load(f)
        else:
            with open(output_file2) as f:
                actual = json.load(f)
        self.assertEqual(expected, gtest_json_test_utils.normalize(actual))
if __name__ == '__main__':
    os.environ['GTEST_STACK_TRACE_DEPTH'] = '0'
    gtest_test_utils.Main()