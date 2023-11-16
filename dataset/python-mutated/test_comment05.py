from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('comment05.xlsx')
        self.ignore_files = ['xl/drawings/vmlDrawing1.vml']

    def test_create_file(self):
        if False:
            print('Hello World!')
        '\n        Test the creation of a simple XlsxWriter file with comments.\n        Test the VML data and shape ids for blocks of comments > 1024.\n        '
        workbook = Workbook(self.got_filename)
        worksheet1 = workbook.add_worksheet()
        worksheet2 = workbook.add_worksheet()
        worksheet3 = workbook.add_worksheet()
        for row in range(0, 127 + 1):
            for col in range(0, 15 + 1):
                worksheet1.write_comment(row, col, 'Some text')
        worksheet3.write_comment('A1', 'More text')
        worksheet1.set_comments_author('John')
        worksheet3.set_comments_author('John')
        workbook.close()
        self.assertExcelEqual()