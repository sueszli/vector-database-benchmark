from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('comment06.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with comments.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_comment('A1', 'Some text')
        worksheet.write_comment('A2', 'Some text')
        worksheet.write_comment('A3', 'Some text', {'visible': True})
        worksheet.write_comment('A4', 'Some text')
        worksheet.write_comment('A5', 'Some text')
        worksheet.set_comments_author('John')
        workbook.close()
        self.assertExcelEqual()