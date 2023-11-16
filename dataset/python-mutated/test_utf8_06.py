from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('utf8_06.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of an XlsxWriter file with utf-8 strings.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        bold = workbook.add_format({'bold': 1})
        italic = workbook.add_format({'italic': 1})
        worksheet.write('A1', 'Foo', bold)
        worksheet.write('A2', 'Bar', italic)
        worksheet.write_rich_string('A3', 'Caf', bold, 'é')
        workbook.close()
        self.assertExcelEqual()