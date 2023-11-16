from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('types04.xlsx')

    def test_write_url_default(self):
        if False:
            while True:
                i = 10
        'Test writing hyperlinks with strings_to_urls on.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        red = workbook.add_format({'font_color': 'red'})
        worksheet.write(0, 0, 'http://www.google.com/', red)
        worksheet.write_string(1, 0, 'http://www.google.com/', red)
        workbook.close()
        self.assertExcelEqual()

    def test_write_url_implicit(self):
        if False:
            i = 10
            return i + 15
        'Test writing hyperlinks with strings_to_urls on.'
        workbook = Workbook(self.got_filename, {'strings_to_urls': True})
        worksheet = workbook.add_worksheet()
        red = workbook.add_format({'font_color': 'red'})
        worksheet.write(0, 0, 'http://www.google.com/', red)
        worksheet.write_string(1, 0, 'http://www.google.com/', red)
        workbook.close()
        self.assertExcelEqual()

    def test_write_url_explicit(self):
        if False:
            return 10
        'Test writing hyperlinks with strings_to_urls off.'
        workbook = Workbook(self.got_filename, {'strings_to_urls': False})
        worksheet = workbook.add_worksheet()
        red = workbook.add_format({'font_color': 'red'})
        worksheet.write_url(0, 0, 'http://www.google.com/', red)
        worksheet.write(1, 0, 'http://www.google.com/', red)
        workbook.close()
        self.assertExcelEqual()