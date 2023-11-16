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
        self.set_filename('hyperlink01.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with hyperlinks'
        workbook = Workbook(self.got_filename)
        workbook.default_url_format = None
        worksheet = workbook.add_worksheet()
        worksheet.write_url('A1', 'http://www.perl.org/')
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_write(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with hyperlinks with write()'
        workbook = Workbook(self.got_filename)
        workbook.default_url_format = None
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'http://www.perl.org/')
        workbook.close()
        self.assertExcelEqual()