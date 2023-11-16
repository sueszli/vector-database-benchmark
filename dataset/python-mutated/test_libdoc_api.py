from io import StringIO
import sys
import tempfile
import unittest
from robot import libdoc
from robot.utils.asserts import assert_equal

class TestLibdoc(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        sys.stdout = StringIO()

    def tearDown(self):
        if False:
            return 10
        sys.stdout = sys.__stdout__

    def test_html(self):
        if False:
            while True:
                i = 10
        output = tempfile.mkstemp(suffix='.html')[1]
        libdoc.libdoc('String', output)
        assert_equal(sys.stdout.getvalue().strip(), output)
        with open(output) as f:
            assert '"name": "String"' in f.read()

    def test_xml(self):
        if False:
            while True:
                i = 10
        output = tempfile.mkstemp(suffix='.xml')[1]
        libdoc.libdoc('String', output)
        assert_equal(sys.stdout.getvalue().strip(), output)
        with open(output) as f:
            assert 'name="String"' in f.read()

    def test_format(self):
        if False:
            return 10
        output = tempfile.mkstemp()[1]
        libdoc.libdoc('String', output, format='xml')
        assert_equal(sys.stdout.getvalue().strip(), output)
        with open(output) as f:
            assert 'name="String"' in f.read()

    def test_quiet(self):
        if False:
            print('Hello World!')
        output = tempfile.mkstemp(suffix='.html')[1]
        libdoc.libdoc('String', output, quiet=True)
        assert_equal(sys.stdout.getvalue().strip(), '')
        with open(output) as f:
            assert '"name": "String"' in f.read()

    def test_LibraryDocumentation(self):
        if False:
            return 10
        doc = libdoc.LibraryDocumentation('OperatingSystem')
        assert_equal(doc.name, 'OperatingSystem')
if __name__ == '__main__':
    unittest.main()