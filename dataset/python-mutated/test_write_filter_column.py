import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWriteFilterColumn(unittest.TestCase):
    """
    Test the Worksheet _write_filter_column() method.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_filter_column(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_filter_column() method'
        self.worksheet._write_filter_column(0, 1, ['East'])
        exp = '<filterColumn colId="0"><filters><filter val="East"/></filters></filterColumn>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)