from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_filename('set_row03.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_row(0, 0.75)
        worksheet.set_row(1, 1.5)
        worksheet.set_row(2, 2.25)
        worksheet.set_row(3, 3)
        worksheet.set_row(11, 9)
        worksheet.set_row(12, 9.75)
        worksheet.set_row(13, 10.5)
        worksheet.set_row(14, 11.25)
        worksheet.set_row(18, 14.25)
        worksheet.set_row(20, 15.75, None, {'hidden': True})
        worksheet.set_row(21, 16.5)
        workbook.close()
        self.assertExcelEqual()