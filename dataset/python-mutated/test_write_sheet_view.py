import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWriteSheetView(unittest.TestCase):
    """
    Test the Worksheet _write_sheet_view() method.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_sheet_view_tab_not_selected(self):
        if False:
            print('Hello World!')
        'Test the _write_sheet_view() method. Tab not selected'
        self.worksheet._write_sheet_view()
        exp = '<sheetView workbookViewId="0"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_view_tab_selected(self):
        if False:
            return 10
        'Test the _write_sheet_view() method. Tab selected'
        self.worksheet.select()
        self.worksheet._write_sheet_view()
        exp = '<sheetView tabSelected="1" workbookViewId="0"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_view_hide_gridlines(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_sheet_view() method. Tab selected + hide_gridlines()'
        self.worksheet.select()
        self.worksheet.hide_gridlines()
        self.worksheet._write_sheet_view()
        exp = '<sheetView tabSelected="1" workbookViewId="0"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_view_hide_gridlines_0(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_sheet_view() method. Tab selected + hide_gridlines(0)'
        self.worksheet.select()
        self.worksheet.hide_gridlines(0)
        self.worksheet._write_sheet_view()
        exp = '<sheetView tabSelected="1" workbookViewId="0"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_view_hide_gridlines_1(self):
        if False:
            print('Hello World!')
        'Test the _write_sheet_view() method. Tab selected + hide_gridlines(1)'
        self.worksheet.select()
        self.worksheet.hide_gridlines(1)
        self.worksheet._write_sheet_view()
        exp = '<sheetView tabSelected="1" workbookViewId="0"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_view_hide_gridlines_2(self):
        if False:
            while True:
                i = 10
        'Test the _write_sheet_view() method. Tab selected + hide_gridlines(2)'
        self.worksheet.select()
        self.worksheet.hide_gridlines(2)
        self.worksheet._write_sheet_view()
        exp = '<sheetView showGridLines="0" tabSelected="1" workbookViewId="0"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_view_hide_row_col_headers(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _write_sheet_views() method'
        self.worksheet.select()
        self.worksheet.hide_row_col_headers()
        self.worksheet._write_sheet_view()
        exp = '<sheetView showRowColHeaders="0" tabSelected="1" workbookViewId="0"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)