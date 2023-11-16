import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWriteCustomFilter(unittest.TestCase):
    """
    Test the Worksheet _write_custom_filter() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_custom_filter(self):
        if False:
            while True:
                i = 10
        'Test the _write_custom_filter() method'
        self.worksheet._write_custom_filter(4, 3000)
        exp = '<customFilter operator="greaterThan" val="3000"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)