import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWriteSheetFormatPr(unittest.TestCase):
    """
    Test the Worksheet _write_sheet_format_pr() method.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_sheet_format_pr(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _write_sheet_format_pr() method'
        self.worksheet._write_sheet_format_pr()
        exp = '<sheetFormatPr defaultRowHeight="15"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)