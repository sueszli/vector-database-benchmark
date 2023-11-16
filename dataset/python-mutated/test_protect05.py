from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('protect05.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the a simple XlsxWriter file with worksheet protection.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        unlocked = workbook.add_format({'locked': 0, 'hidden': 0})
        hidden = workbook.add_format({'locked': 0, 'hidden': 1})
        worksheet.protect()
        worksheet.unprotect_range('=A1')
        worksheet.unprotect_range('$C$1:$C$3')
        worksheet.unprotect_range('G4:I6', 'MyRange')
        worksheet.unprotect_range('K7')
        worksheet.write('A1', 1)
        worksheet.write('A2', 2, unlocked)
        worksheet.write('A3', 3, hidden)
        workbook.close()
        self.assertExcelEqual()