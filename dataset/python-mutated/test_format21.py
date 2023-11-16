from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('format21.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file with automatic color.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        format1 = workbook.add_format({'color': 'automatic', 'fg_color': 'automatic', 'bg_color': 'red', 'pattern': 6})
        worksheet.write(0, 0, 'Foo', format1)
        workbook.close()
        self.assertExcelEqual()