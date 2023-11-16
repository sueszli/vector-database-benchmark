from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_filename('default_row05.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_default_row(24, 1)
        worksheet.write('A1', 'Foo')
        worksheet.write('A10', 'Bar')
        worksheet.write('A20', 'Baz')
        for row in range(1, 8 + 1):
            worksheet.set_row(row, 24)
        for row in range(10, 19 + 1):
            worksheet.set_row(row, 24)
        workbook.close()
        self.assertExcelEqual()