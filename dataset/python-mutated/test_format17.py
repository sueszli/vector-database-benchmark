from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_filename('format17.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with a pattern only.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        pattern = workbook.add_format({'pattern': 2, 'fg_color': 'red'})
        worksheet.write('A1', '', pattern)
        workbook.close()
        self.assertExcelEqual()