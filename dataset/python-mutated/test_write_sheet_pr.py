import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWriteSheetPr(unittest.TestCase):
    """
    Test the Worksheet _write_sheet_pr() method.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_sheet_pr_fit_to_page(self):
        if False:
            return 10
        'Test the _write_sheet_pr() method'
        self.worksheet.fit_to_pages(1, 1)
        self.worksheet._write_sheet_pr()
        exp = '<sheetPr><pageSetUpPr fitToPage="1"/></sheetPr>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_pr_tab_color(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _write_sheet_pr() method'
        self.worksheet.set_tab_color('red')
        self.worksheet._write_sheet_pr()
        exp = '<sheetPr><tabColor rgb="FFFF0000"/></sheetPr>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_pr_both(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_sheet_pr() method'
        self.worksheet.set_tab_color('red')
        self.worksheet.fit_to_pages(1, 1)
        self.worksheet._write_sheet_pr()
        exp = '<sheetPr><tabColor rgb="FFFF0000"/><pageSetUpPr fitToPage="1"/></sheetPr>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)