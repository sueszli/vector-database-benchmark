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
        self.set_filename('hyperlink30.xlsx')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with hyperlinks.'
        workbook = Workbook(self.got_filename)
        workbook.custom_colors = ['FF0000FF']
        worksheet = workbook.add_worksheet()
        format1 = workbook.add_format({'hyperlink': 1})
        format2 = workbook.add_format({'color': 'red', 'underline': 1})
        format3 = workbook.add_format({'color': 'blue', 'underline': 1})
        worksheet.write_url('A1', 'http://www.python.org/1', format1)
        worksheet.write_url('A2', 'http://www.python.org/2', format2)
        worksheet.write_url('A3', 'http://www.python.org/3', format3)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_with_default_format(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with hyperlinks.'
        workbook = Workbook(self.got_filename)
        workbook.custom_colors = ['FF0000FF']
        worksheet = workbook.add_worksheet()
        format2 = workbook.add_format({'color': 'red', 'underline': 1})
        format3 = workbook.add_format({'color': 'blue', 'underline': 1})
        worksheet.write_url('A1', 'http://www.python.org/1')
        worksheet.write_url('A2', 'http://www.python.org/2', format2)
        worksheet.write_url('A3', 'http://www.python.org/3', format3)
        workbook.close()
        self.assertExcelEqual()