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
        self.set_filename('table22.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with tables.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        data = [['apple', 'pie'], ['pine', 'tree']]
        worksheet.set_column('B:C', 10.288)
        worksheet.add_table('B2:C3', {'data': data, 'header_row': False})
        workbook.close()
        self.assertExcelEqual()