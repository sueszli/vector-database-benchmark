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
        self.set_filename('default_format01.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename, {'default_format_properties': {'font_size': 10}})
        worksheet = workbook.add_worksheet()
        worksheet.set_default_row(12.75)
        worksheet.original_row_height = 12.75
        workbook.close()
        self.assertExcelEqual()