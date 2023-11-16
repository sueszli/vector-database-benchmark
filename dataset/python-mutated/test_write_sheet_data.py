import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWriteSheetData(unittest.TestCase):
    """
    Test the Worksheet _write_sheet_data() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_sheet_data(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_sheet_data() method'
        self.worksheet._write_sheet_data()
        exp = '<sheetData/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)