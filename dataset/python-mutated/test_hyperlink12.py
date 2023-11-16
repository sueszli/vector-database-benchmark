from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_filename('hyperlink12.xlsx')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with hyperlinks. This example has link formatting.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        format = workbook.add_format({'color': 'blue', 'underline': 1})
        worksheet.write_url('A1', 'mailto:jmcnamara@cpan.org', format)
        worksheet.write_url('A3', 'ftp://perl.org/', format)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_write(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with hyperlinks. This example has link formatting and uses write()'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        format = workbook.add_format({'color': 'blue', 'underline': 1})
        worksheet.write('A1', 'mailto:jmcnamara@cpan.org', format)
        worksheet.write('A3', 'ftp://perl.org/', format)
        workbook.close()
        self.assertExcelEqual()