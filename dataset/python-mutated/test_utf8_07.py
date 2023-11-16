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
        self.set_filename('utf8_07.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of an XlsxWriter file with utf-8 strings.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Foo')
        worksheet.write_comment('A1', 'Café')
        worksheet.set_comments_author('John')
        workbook.close()
        self.assertExcelEqual()