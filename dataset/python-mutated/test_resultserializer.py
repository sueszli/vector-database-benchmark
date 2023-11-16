import unittest
from io import BytesIO, StringIO
from robot.result import ExecutionResult
from robot.reporting.outputwriter import OutputWriter
from robot.utils import ET, ETSource, XmlWriter
from robot.utils.asserts import assert_equal
from test_resultbuilder import GOLDEN_XML, GOLDEN_XML_TWICE

class StreamXmlWriter(XmlWriter):

    def _create_output(self, output):
        if False:
            for i in range(10):
                print('nop')
        return output

    def close(self):
        if False:
            while True:
                i = 10
        pass

class TestableOutputWriter(OutputWriter):

    def _get_writer(self, output, rpa, generator):
        if False:
            print('Hello World!')
        writer = StreamXmlWriter(output, write_empty=False)
        writer.start('robot')
        return writer

class TestResultSerializer(unittest.TestCase):

    def test_single_result_serialization(self):
        if False:
            return 10
        output = StringIO()
        writer = TestableOutputWriter(output)
        ExecutionResult(GOLDEN_XML).visit(writer)
        self._assert_xml_content(self._xml_lines(output.getvalue()), self._xml_lines(GOLDEN_XML))

    def _xml_lines(self, text):
        if False:
            i = 10
            return i + 15
        with ETSource(text) as source:
            tree = ET.parse(source)
        output = BytesIO()
        tree.write(output)
        return output.getvalue().splitlines()

    def _assert_xml_content(self, actual, expected):
        if False:
            while True:
                i = 10
        assert_equal(len(actual), len(expected))
        for (index, (act, exp)) in enumerate(list(zip(actual, expected))[2:]):
            assert_equal(act, exp.strip(), 'Different values on line %d' % index)

    def test_combining_results(self):
        if False:
            while True:
                i = 10
        output = StringIO()
        writer = TestableOutputWriter(output)
        ExecutionResult(GOLDEN_XML, GOLDEN_XML).visit(writer)
        self._assert_xml_content(self._xml_lines(output.getvalue()), self._xml_lines(GOLDEN_XML_TWICE))
if __name__ == '__main__':
    unittest.main()