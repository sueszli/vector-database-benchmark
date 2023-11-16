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
        self.set_filename('rich_string09.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        bold = workbook.add_format({'bold': 1})
        italic = workbook.add_format({'italic': 1})
        worksheet.write('A1', 'Foo', bold)
        worksheet.write('A2', 'Bar', italic)
        worksheet.write_rich_string('A3', 'a', bold, 'bc', 'defg')
        import warnings
        warnings.filterwarnings('ignore')
        worksheet.write_rich_string('A3', 'a', bold, bold, 'bc', 'defg')
        worksheet.write_rich_string('A3', '', bold, 'bc', 'defg')
        worksheet.write_rich_string('A3', 'a', bold, '', 'defg')
        worksheet.write_rich_string('A3', 'a', bold, 'bc', '')
        worksheet.write_rich_string('A3', 'a')
        worksheet.write_rich_string('A3', 'a', bold)
        worksheet.write_rich_string('A3', 'a', bold, italic)
        workbook.close()
        self.assertExcelEqual()