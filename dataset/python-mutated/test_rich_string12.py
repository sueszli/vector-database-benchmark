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
        self.set_filename('rich_string12.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column('A:A', 30)
        worksheet.set_row(2, 60)
        bold = workbook.add_format({'bold': 1})
        italic = workbook.add_format({'italic': 1})
        wrap = workbook.add_format({'text_wrap': 1})
        worksheet.write('A1', 'Foo', bold)
        worksheet.write('A2', 'Bar', italic)
        worksheet.write_rich_string('A3', 'This is\n', bold, 'bold\n', 'and this is\n', italic, 'italic', wrap)
        workbook.close()
        self.assertExcelEqual()