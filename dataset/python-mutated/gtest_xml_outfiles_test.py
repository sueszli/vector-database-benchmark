"""Unit test for the gtest_xml_output module."""
import os
from xml.dom import minidom, Node
import gtest_test_utils
import gtest_xml_test_utils
GTEST_OUTPUT_SUBDIR = 'xml_outfiles'
GTEST_OUTPUT_1_TEST = 'gtest_xml_outfile1_test_'
GTEST_OUTPUT_2_TEST = 'gtest_xml_outfile2_test_'
EXPECTED_XML_1 = '<?xml version="1.0" encoding="UTF-8"?>\n<testsuites tests="1" failures="0" disabled="0" errors="0" time="*" timestamp="*" name="AllTests">\n  <testsuite name="PropertyOne" tests="1" failures="0" disabled="0" errors="0" time="*">\n    <testcase name="TestSomeProperties" status="run" result="completed" time="*" classname="PropertyOne">\n      <properties>\n        <property name="SetUpProp" value="1"/>\n        <property name="TestSomeProperty" value="1"/>\n        <property name="TearDownProp" value="1"/>\n      </properties>\n    </testcase>\n  </testsuite>\n</testsuites>\n'
EXPECTED_XML_2 = '<?xml version="1.0" encoding="UTF-8"?>\n<testsuites tests="1" failures="0" disabled="0" errors="0" time="*" timestamp="*" name="AllTests">\n  <testsuite name="PropertyTwo" tests="1" failures="0" disabled="0" errors="0" time="*">\n    <testcase name="TestSomeProperties" status="run" result="completed" time="*" classname="PropertyTwo">\n      <properties>\n        <property name="SetUpProp" value="2"/>\n        <property name="TestSomeProperty" value="2"/>\n        <property name="TearDownProp" value="2"/>\n      </properties>\n    </testcase>\n  </testsuite>\n</testsuites>\n'

class GTestXMLOutFilesTest(gtest_xml_test_utils.GTestXMLTestCase):
    """Unit test for Google Test's XML output functionality."""

    def setUp(self):
        if False:
            print('Hello World!')
        self.output_dir_ = os.path.join(gtest_test_utils.GetTempDir(), GTEST_OUTPUT_SUBDIR, '')
        self.DeleteFilesAndDir()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.DeleteFilesAndDir()

    def DeleteFilesAndDir(self):
        if False:
            print('Hello World!')
        try:
            os.remove(os.path.join(self.output_dir_, GTEST_OUTPUT_1_TEST + '.xml'))
        except os.error:
            pass
        try:
            os.remove(os.path.join(self.output_dir_, GTEST_OUTPUT_2_TEST + '.xml'))
        except os.error:
            pass
        try:
            os.rmdir(self.output_dir_)
        except os.error:
            pass

    def testOutfile1(self):
        if False:
            while True:
                i = 10
        self._TestOutFile(GTEST_OUTPUT_1_TEST, EXPECTED_XML_1)

    def testOutfile2(self):
        if False:
            print('Hello World!')
        self._TestOutFile(GTEST_OUTPUT_2_TEST, EXPECTED_XML_2)

    def _TestOutFile(self, test_name, expected_xml):
        if False:
            i = 10
            return i + 15
        gtest_prog_path = gtest_test_utils.GetTestExecutablePath(test_name)
        command = [gtest_prog_path, '--gtest_output=xml:%s' % self.output_dir_]
        p = gtest_test_utils.Subprocess(command, working_dir=gtest_test_utils.GetTempDir())
        self.assert_(p.exited)
        self.assertEquals(0, p.exit_code)
        output_file_name1 = test_name + '.xml'
        output_file1 = os.path.join(self.output_dir_, output_file_name1)
        output_file_name2 = 'lt-' + output_file_name1
        output_file2 = os.path.join(self.output_dir_, output_file_name2)
        self.assert_(os.path.isfile(output_file1) or os.path.isfile(output_file2), output_file1)
        expected = minidom.parseString(expected_xml)
        if os.path.isfile(output_file1):
            actual = minidom.parse(output_file1)
        else:
            actual = minidom.parse(output_file2)
        self.NormalizeXml(actual.documentElement)
        self.AssertEquivalentNodes(expected.documentElement, actual.documentElement)
        expected.unlink()
        actual.unlink()
if __name__ == '__main__':
    os.environ['GTEST_STACK_TRACE_DEPTH'] = '0'
    gtest_test_utils.Main()