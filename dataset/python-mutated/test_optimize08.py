from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_filename('optimize08.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename, {'constant_memory': True, 'in_memory': False})
        worksheet = workbook.add_worksheet()
        bold = workbook.add_format({'bold': 1})
        italic = workbook.add_format({'italic': 1})
        worksheet.write('A1', 'Foo', bold)
        worksheet.write('A2', 'Bar', italic)
        worksheet.write_rich_string('A3', ' a', bold, 'bc', 'defg ')
        workbook.close()
        self.assertExcelEqual()