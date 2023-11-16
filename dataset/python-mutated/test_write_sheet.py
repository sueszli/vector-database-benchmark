import unittest
from io import StringIO
from ...workbook import Workbook

class TestWriteSheet(unittest.TestCase):
    """
    Test the Workbook _write_sheet() method.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.fh = StringIO()
        self.workbook = Workbook()
        self.workbook._set_filehandle(self.fh)

    def test_write_sheet1(self):
        if False:
            return 10
        'Test the _write_sheet() method'
        self.workbook._write_sheet('Sheet1', 1, 0)
        exp = '<sheet name="Sheet1" sheetId="1" r:id="rId1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet2(self):
        if False:
            while True:
                i = 10
        'Test the _write_sheet() method'
        self.workbook._write_sheet('Sheet1', 1, 1)
        exp = '<sheet name="Sheet1" sheetId="1" state="hidden" r:id="rId1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet3(self):
        if False:
            print('Hello World!')
        'Test the _write_sheet() method'
        self.workbook._write_sheet('Bits & Bobs', 1, 0)
        exp = '<sheet name="Bits &amp; Bobs" sheetId="1" r:id="rId1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def tearDown(self):
        if False:
            return 10
        self.workbook.fileclosed = 1