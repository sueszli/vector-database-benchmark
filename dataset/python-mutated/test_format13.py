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
        self.set_filename('format13.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_row(0, 21)
        font_format = workbook.add_format()
        font_format.set_font('B Nazanin')
        font_format.set_font_family(0)
        font_format.set_font_charset(178)
        worksheet.write('A1', 'Foo', font_format)
        workbook.close()
        self.assertExcelEqual()