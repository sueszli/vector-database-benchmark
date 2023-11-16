from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('escapes05.xlsx')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file. Check encoding of url strings.'
        workbook = Workbook(self.got_filename)
        workbook.default_url_format = None
        worksheet1 = workbook.add_worksheet('Start')
        worksheet2 = workbook.add_worksheet('A & B')
        worksheet1.write_url('A1', "internal:'A & B'!A1", None, 'Jump to A & B')
        workbook.close()
        self.assertExcelEqual()