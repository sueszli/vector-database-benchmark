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
        self.set_filename('default_row02.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_default_row(hide_unused_rows=True)
        worksheet.write('A1', 'Foo')
        worksheet.write('A10', 'Bar')
        for row in range(1, 8 + 1):
            worksheet.set_row(row, 15)
        workbook.close()
        self.assertExcelEqual()