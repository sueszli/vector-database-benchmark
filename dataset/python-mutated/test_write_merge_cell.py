import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWriteMergeCell(unittest.TestCase):
    """
    Test the Worksheet _write_merge_cell() method.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_merge_cell(self):
        if False:
            return 10
        'Test the _write_merge_cell() method'
        self.worksheet._write_merge_cell([2, 1, 2, 2])
        exp = '<mergeCell ref="B3:C3"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)