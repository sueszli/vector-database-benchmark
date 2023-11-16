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
        self.set_filename('hyperlink28.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file with hyperlinks.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        format = workbook.add_format({'hyperlink': True})
        worksheet.write_url('A1', 'http://www.perl.org/', format)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_with_workbook_format(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with hyperlinks.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        format = workbook.get_default_url_format()
        worksheet.write_url('A1', 'http://www.perl.org/', format)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_with_default_format(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with hyperlinks.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_url('A1', 'http://www.perl.org/')
        workbook.close()
        self.assertExcelEqual()