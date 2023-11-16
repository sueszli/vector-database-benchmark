"""Tests for the pdeps script in the Tools directory."""
import os
import unittest
import tempfile
from test.test_tools import skip_if_missing, import_tool
skip_if_missing()

class PdepsTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        self.pdeps = import_tool('pdeps')

    def test_process_errors(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, 'foo')
            with open(fn, 'w') as stream:
                stream.write('#!/this/will/fail')
            self.pdeps.process(fn, {})

    def test_inverse_attribute_error(self):
        if False:
            i = 10
            return i + 15
        self.pdeps.inverse({'a': []})
if __name__ == '__main__':
    unittest.main()