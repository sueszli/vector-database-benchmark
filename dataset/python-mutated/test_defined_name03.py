from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('defined_name03.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file with defined names.'
        workbook = Workbook(self.got_filename)
        worksheet1 = workbook.add_worksheet('sheet One')
        workbook.define_name('Sales', "='sheet One'!G1:H10")
        workbook.close()
        self.assertExcelEqual()