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
        self.set_filename('comment13.xlsx')
        self.ignore_files = ['xl/styles.xml']

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with comments.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Foo')
        worksheet.write_comment('B2', 'Some text', {'font_name': 'Courier', 'font_size': 10, 'font_family': 3})
        worksheet.set_comments_author('John')
        workbook.close()
        self.assertExcelEqual()