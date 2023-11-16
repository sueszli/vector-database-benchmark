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
        self.set_filename('excel2003_style04.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename, {'excel2003_style': True})
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Foo')
        worksheet.set_row(0, 21)
        workbook.close()
        self.assertExcelEqual()