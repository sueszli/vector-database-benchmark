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
        self.set_filename('simple09.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write('A0', 'foo')
        worksheet.write(-1, -1, 'foo')
        worksheet.write(0, -1, 'foo')
        worksheet.write(-1, 0, 'foo')
        worksheet.write(1048576, 0, 'foo')
        worksheet.write(0, 16384, 'foo')
        workbook.close()
        self.assertExcelEqual()