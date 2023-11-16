import unittest
from io import StringIO
from ...workbook import Workbook

class TestAssembleWorkbook(unittest.TestCase):
    """
    Test assembling a complete Workbook file.

    """

    def test_get_worksheet_by_name01(self):
        if False:
            i = 10
            return i + 15
        'Test get_worksheet_by_name()'
        fh = StringIO()
        workbook = Workbook()
        workbook._set_filehandle(fh)
        exp = workbook.add_worksheet()
        got = workbook.get_worksheet_by_name('Sheet1')
        workbook.fileclosed = 1
        self.assertEqual(got, exp)

    def test_get_worksheet_by_name02(self):
        if False:
            print('Hello World!')
        'Test get_worksheet_by_name()'
        fh = StringIO()
        workbook = Workbook()
        workbook._set_filehandle(fh)
        workbook.add_worksheet()
        exp = workbook.add_worksheet()
        got = workbook.get_worksheet_by_name('Sheet2')
        workbook.fileclosed = 1
        self.assertEqual(got, exp)

    def test_get_worksheet_by_name03(self):
        if False:
            while True:
                i = 10
        'Test get_worksheet_by_name()'
        fh = StringIO()
        workbook = Workbook()
        workbook._set_filehandle(fh)
        exp = workbook.add_worksheet('Sheet 3')
        got = workbook.get_worksheet_by_name('Sheet 3')
        workbook.fileclosed = 1
        self.assertEqual(got, exp)

    def test_get_worksheet_by_name04(self):
        if False:
            print('Hello World!')
        'Test get_worksheet_by_name()'
        fh = StringIO()
        workbook = Workbook()
        workbook._set_filehandle(fh)
        exp = workbook.add_worksheet("Sheet '4")
        got = workbook.get_worksheet_by_name("Sheet '4")
        workbook.fileclosed = 1
        self.assertEqual(got, exp)

    def test_get_worksheet_by_name05(self):
        if False:
            while True:
                i = 10
        'Test get_worksheet_by_name()'
        fh = StringIO()
        workbook = Workbook()
        workbook._set_filehandle(fh)
        exp = None
        got = workbook.get_worksheet_by_name('Sheet 5')
        workbook.fileclosed = 1
        self.assertEqual(got, exp)