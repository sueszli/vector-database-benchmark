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
        self.set_filename('hyperlink22.xlsx')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with hyperlinks.'
        workbook = Workbook(self.got_filename)
        workbook.default_url_format = None
        worksheet = workbook.add_worksheet()
        worksheet.write_url('A1', 'external:\\\\Vboxsvr\\share\\foo bar.xlsx')
        workbook.close()
        self.assertExcelEqual()