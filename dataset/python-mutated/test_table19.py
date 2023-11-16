from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('table19.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file with tables.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column('C:F', 10.288)
        worksheet.add_table('C3:F13', {'columns': [{}, {}, {}, {'header': ' Column4 '}]})
        workbook.close()
        self.assertExcelEqual()