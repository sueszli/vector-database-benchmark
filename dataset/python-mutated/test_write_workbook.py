import unittest
from io import StringIO
from ...workbook import Workbook

class TestWriteWorkbook(unittest.TestCase):
    """
    Test the Workbook _write_workbook() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.workbook = Workbook()
        self.workbook._set_filehandle(self.fh)

    def test_write_workbook(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_workbook() method'
        self.workbook._write_workbook()
        exp = '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.workbook.fileclosed = 1