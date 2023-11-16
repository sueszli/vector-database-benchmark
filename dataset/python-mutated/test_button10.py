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
        self.set_filename('button10.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet1 = workbook.add_worksheet()
        worksheet2 = workbook.add_worksheet()
        worksheet3 = workbook.add_worksheet()
        worksheet1.write_comment('A1', 'Some text')
        worksheet2.insert_button('B2', {})
        worksheet3.write_comment('C2', 'More text')
        worksheet1.set_comments_author('John')
        worksheet3.set_comments_author('John')
        workbook.close()
        self.assertExcelEqual()