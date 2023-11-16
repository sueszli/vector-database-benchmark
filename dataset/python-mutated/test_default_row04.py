from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('default_row04.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_default_row(24)
        worksheet.write('A1', 'Foo')
        worksheet.write('A10', 'Bar')
        worksheet.write_comment('C4', 'Hello', {'y_offset': 22})
        worksheet.set_comments_author('John')
        workbook.close()
        self.assertExcelEqual()