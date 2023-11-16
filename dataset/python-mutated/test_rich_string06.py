from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('rich_string06.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        red = workbook.add_format({'color': 'red'})
        worksheet.write('A1', 'Foo', red)
        worksheet.write('A2', 'Bar')
        worksheet.write_rich_string('A3', 'ab', red, 'cde', 'fg')
        workbook.close()
        self.assertExcelEqual()