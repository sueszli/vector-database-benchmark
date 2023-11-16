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
        self.set_filename('comment03.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with comments.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Foo')
        worksheet.write_comment('A1', 'Some text')
        worksheet.write_comment('XFD1048576', 'Some text')
        worksheet.set_comments_author('John')
        workbook.close()
        self.assertExcelEqual()