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
        self.set_filename('autofit13.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_string(0, 0, 'Foo')
        worksheet.write_string(0, 1, 'Foo bar')
        worksheet.write_string(0, 2, 'Foo bar bar')
        worksheet.autofilter(0, 0, 0, 2)
        worksheet.autofit()
        workbook.close()
        self.assertExcelEqual()