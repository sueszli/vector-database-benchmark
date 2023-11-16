from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_filename('autofit08.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_string(0, 0, 'a')
        worksheet.write_string(1, 0, 'aaa')
        worksheet.write_string(2, 0, 'a')
        worksheet.write_string(3, 0, 'aaaa')
        worksheet.write_string(4, 0, 'a')
        worksheet.autofit()
        workbook.close()
        self.assertExcelEqual()