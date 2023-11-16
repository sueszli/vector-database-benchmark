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
        self.set_filename('protect07.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the a simple XlsxWriter file with "read-only recommended".'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        workbook.read_only_recommended()
        workbook.close()
        self.assertExcelEqual()