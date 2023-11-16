import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWriteSheetViews(unittest.TestCase):
    """
    Test the Worksheet _write_sheet_views() method.
    With explicit top/left cells.

    """

    def setUp(self):
        if False:
            return 10
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_sheet_views1(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _write_sheet_views() method with split panes + selection'
        self.worksheet.select()
        self.worksheet.set_selection('A2')
        self.worksheet.split_panes(15, 0, 20, 0)
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" workbookViewId="0"><pane ySplit="600" topLeftCell="A21" activePane="bottomLeft"/><selection pane="bottomLeft" activeCell="A2" sqref="A2"/></sheetView></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views2(self):
        if False:
            return 10
        'Test the _write_sheet_views() method with split panes + selection'
        self.worksheet.select()
        self.worksheet.set_selection('A21')
        self.worksheet.split_panes(15, 0, 20, 0)
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" workbookViewId="0"><pane ySplit="600" topLeftCell="A21" activePane="bottomLeft"/><selection pane="bottomLeft" activeCell="A21" sqref="A21"/></sheetView></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views3(self):
        if False:
            return 10
        'Test the _write_sheet_views() method with split panes + selection'
        self.worksheet.select()
        self.worksheet.set_selection('B1')
        self.worksheet.split_panes(0, 8.43, 0, 4)
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" workbookViewId="0"><pane xSplit="1350" topLeftCell="E1" activePane="topRight"/><selection pane="topRight" activeCell="B1" sqref="B1"/></sheetView></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views4(self):
        if False:
            while True:
                i = 10
        'Test the _write_sheet_views() method with split panes + selection'
        self.worksheet.select()
        self.worksheet.set_selection('E1')
        self.worksheet.split_panes(0, 8.43, 0, 4)
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" workbookViewId="0"><pane xSplit="1350" topLeftCell="E1" activePane="topRight"/><selection pane="topRight" activeCell="E1" sqref="E1"/></sheetView></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)