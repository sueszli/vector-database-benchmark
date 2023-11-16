import unittest
from xlsxwriter.chartsheet import Chartsheet
from xlsxwriter.worksheet import Worksheet
from ...workbook import Workbook

class MyWorksheet(Worksheet):
    pass

class MyChartsheet(Chartsheet):
    pass

class MyWorkbook(Workbook):
    chartsheet_class = MyChartsheet
    worksheet_class = MyWorksheet

class TestCustomSheet(unittest.TestCase):
    """
    Test the Workbook _check_sheetname() method.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.workbook = Workbook()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.workbook.fileclosed = 1

    def test_check_chartsheet(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _check_sheetname() method'
        sheet = self.workbook.add_chartsheet()
        assert isinstance(sheet, Chartsheet)
        sheet = self.workbook.add_chartsheet(chartsheet_class=MyChartsheet)
        assert isinstance(sheet, MyChartsheet)

    def test_check_worksheet(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _check_sheetname() method'
        sheet = self.workbook.add_worksheet()
        assert isinstance(sheet, Worksheet)
        sheet = self.workbook.add_worksheet(worksheet_class=MyWorksheet)
        assert isinstance(sheet, MyWorksheet)

class TestCustomWorkBook(unittest.TestCase):
    """
    Test the Workbook _check_sheetname() method.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.workbook = MyWorkbook()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.workbook.fileclosed = 1

    def test_check_chartsheet(self):
        if False:
            while True:
                i = 10
        'Test the _check_sheetname() method'
        sheet = self.workbook.add_chartsheet()
        assert isinstance(sheet, MyChartsheet)

    def test_check_worksheet(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _check_sheetname() method'
        sheet = self.workbook.add_worksheet()
        assert isinstance(sheet, MyWorksheet)