from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('protect04.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the a simple XlsxWriter file with worksheet protection.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        unlocked = workbook.add_format({'locked': 0, 'hidden': 0})
        hidden = workbook.add_format({'locked': 0, 'hidden': 1})
        worksheet.protect()
        worksheet.unprotect_range('A1')
        worksheet.write('A1', 1)
        worksheet.write('A2', 2, unlocked)
        worksheet.write('A3', 3, hidden)
        workbook.close()
        self.assertExcelEqual()