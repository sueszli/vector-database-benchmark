import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWriteSheetViews(unittest.TestCase):
    """
    Test the Worksheet _write_sheet_views() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_sheet_views1(self):
        if False:
            return 10
        'Test the _write_sheet_views() method with freeze panes'
        self.worksheet.select()
        self.worksheet.set_selection('A2')
        self.worksheet.freeze_panes(1, 0, 20, 0)
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" workbookViewId="0"><pane ySplit="1" topLeftCell="A21" activePane="bottomLeft" state="frozen"/><selection pane="bottomLeft" activeCell="A2" sqref="A2"/></sheetView></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _write_sheet_views() method with freeze panes'
        self.worksheet.select()
        self.worksheet.set_selection('A1')
        self.worksheet.freeze_panes(1, 0, 20, 0)
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" workbookViewId="0"><pane ySplit="1" topLeftCell="A21" activePane="bottomLeft" state="frozen"/><selection pane="bottomLeft"/></sheetView></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views3(self):
        if False:
            print('Hello World!')
        'Test the _write_sheet_views() method with freeze panes'
        self.worksheet.select()
        self.worksheet.set_selection('B1')
        self.worksheet.freeze_panes(0, 1, 0, 4)
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" workbookViewId="0"><pane xSplit="1" topLeftCell="E1" activePane="topRight" state="frozen"/><selection pane="topRight" activeCell="B1" sqref="B1"/></sheetView></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views4(self):
        if False:
            print('Hello World!')
        'Test the _write_sheet_views() method with freeze panes'
        self.worksheet.select()
        self.worksheet.freeze_panes(0, 1, 0, 4)
        self.worksheet.set_selection('A1')
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" workbookViewId="0"><pane xSplit="1" topLeftCell="E1" activePane="topRight" state="frozen"/><selection pane="topRight"/></sheetView></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views5(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_sheet_views() method with freeze panes'
        self.worksheet.select()
        self.worksheet.set_selection('G4')
        self.worksheet.freeze_panes(3, 6, 6, 8)
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" workbookViewId="0"><pane xSplit="6" ySplit="3" topLeftCell="I7" activePane="bottomRight" state="frozen"/><selection pane="topRight" activeCell="G1" sqref="G1"/><selection pane="bottomLeft" activeCell="A4" sqref="A4"/><selection pane="bottomRight" activeCell="G4" sqref="G4"/></sheetView></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_sheet_views6(self):
        if False:
            while True:
                i = 10
        'Test the _write_sheet_views() method with freeze panes'
        self.worksheet.select()
        self.worksheet.set_selection('A1')
        self.worksheet.freeze_panes(3, 6, 6, 8)
        self.worksheet._write_sheet_views()
        exp = '<sheetViews><sheetView tabSelected="1" workbookViewId="0"><pane xSplit="6" ySplit="3" topLeftCell="I7" activePane="bottomRight" state="frozen"/><selection pane="topRight" activeCell="G1" sqref="G1"/><selection pane="bottomLeft" activeCell="A4" sqref="A4"/><selection pane="bottomRight"/></sheetView></sheetViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)