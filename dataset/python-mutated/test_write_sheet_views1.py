import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWriteSheetViews(unittest.TestCase):
    """
    Test the Worksheet _write_sheet_views() method.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_sheet_views(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_sheet_views() method'
        self.worksheet.select()
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" workbookViewId="0"/></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views_zoom_100(self):
        if False:
            while True:
                i = 10
        'Test the _write_sheet_views() method'
        self.worksheet.select()
        self.worksheet.set_zoom(100)
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" workbookViewId="0"/></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views_zoom_200(self):
        if False:
            return 10
        'Test the _write_sheet_views() method'
        self.worksheet.select()
        self.worksheet.set_zoom(200)
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" zoomScale="200" zoomScaleNormal="200" workbookViewId="0"/></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views_right_to_left(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _write_sheet_views() method'
        self.worksheet.select()
        self.worksheet.right_to_left()
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView rightToLeft="1" tabSelected="1" workbookViewId="0"/></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views_hide_zero(self):
        if False:
            return 10
        'Test the _write_sheet_views() method'
        self.worksheet.select()
        self.worksheet.hide_zero()
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView showZeros="0" tabSelected="1" workbookViewId="0"/></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views_page_view(self):
        if False:
            print('Hello World!')
        'Test the _write_sheet_views() method'
        self.worksheet.select()
        self.worksheet.set_page_view()
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" view="pageLayout" workbookViewId="0"/></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)