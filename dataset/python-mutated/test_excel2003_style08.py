from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('excel2003_style08.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename, {'excel2003_style': True})
        worksheet = workbook.add_worksheet()
        courier = workbook.add_format({'font_name': 'Courier', 'font_size': 8, 'font_family': 3})
        worksheet.write('A1', 'Foo')
        worksheet.write('A2', 'Bar', courier)
        workbook.close()
        self.assertExcelEqual()