import unittest
from io import StringIO
from ...workbook import Workbook

class TestWriteSheets(unittest.TestCase):
    """
    Test the Workbook _write_sheets() method.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fh = StringIO()
        self.workbook = Workbook()
        self.workbook._set_filehandle(self.fh)

    def test_write_sheets(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_sheets() method'
        self.workbook.add_worksheet('Sheet2')
        self.workbook._write_sheets()
        exp = '<sheets><sheet name="Sheet2" sheetId="1" r:id="rId1"/></sheets>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.workbook.fileclosed = 1