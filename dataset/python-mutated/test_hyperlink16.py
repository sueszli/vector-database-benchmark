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
        self.set_filename('hyperlink16.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        "Test the creation of a simple XlsxWriter file with hyperlinks.This example doesn't have any link formatting and tests the relationship linkage code."
        workbook = Workbook(self.got_filename)
        workbook.default_url_format = None
        worksheet = workbook.add_worksheet()
        worksheet.write_url('B2', 'external:./subdir/blank.xlsx')
        workbook.close()
        self.assertExcelEqual()