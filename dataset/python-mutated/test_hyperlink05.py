from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('hyperlink05.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with hyperlinks.'
        workbook = Workbook(self.got_filename)
        workbook.default_url_format = None
        worksheet = workbook.add_worksheet()
        worksheet.write_url('A1', 'http://www.perl.org/')
        worksheet.write_url('A3', 'http://www.perl.org/', None, 'Perl home')
        worksheet.write_url('A5', 'http://www.perl.org/', None, 'Perl home', 'Tool Tip')
        worksheet.write_url('A7', 'http://www.cpan.org/', None, 'CPAN', 'Download')
        workbook.close()
        self.assertExcelEqual()