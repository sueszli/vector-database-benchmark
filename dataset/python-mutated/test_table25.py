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
        self.set_filename('table25.xlsx')

    def test_create_file_style_is_none(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with tables.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column('C:F', 10.288)
        worksheet.add_table('C3:F13', {'style': None})
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_style_is_blank(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file with tables.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column('C:F', 10.288)
        worksheet.add_table('C3:F13', {'style': ''})
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_style_is_none_str(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with tables.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column('C:F', 10.288)
        worksheet.add_table('C3:F13', {'style': 'None'})
        workbook.close()
        self.assertExcelEqual()