from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_filename('comment09.xlsx')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with comments.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_comment('A1', 'Some text', {'author': 'John'})
        worksheet.write_comment('A2', 'Some text', {'author': 'Perl'})
        worksheet.write_comment('A3', 'Some text')
        worksheet.set_comments_author('John')
        workbook.close()
        self.assertExcelEqual()