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
        self.set_filename('format18.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with a quote prefix.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        quote_prefix = workbook.add_format({'quote_prefix': True})
        worksheet.write_string(0, 0, '= Hello', quote_prefix)
        workbook.close()
        self.assertExcelEqual()