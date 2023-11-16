from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('hyperlink02.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with hyperlinks.'
        workbook = Workbook(self.got_filename)
        workbook.default_url_format = None
        worksheet = workbook.add_worksheet()
        worksheet.write_url('A1', 'http://www.perl.org/')
        worksheet.write_url('D4', 'http://www.perl.org/')
        worksheet.write_url('A8', 'http://www.perl.org/')
        worksheet.write_url('B6', 'http://www.cpan.org/')
        worksheet.write_url('F12', 'http://www.cpan.org/')
        workbook.close()
        self.assertExcelEqual()