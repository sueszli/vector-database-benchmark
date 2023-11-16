from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('set_start_page01.xlsx')
        self.ignore_elements = {'xl/worksheets/sheet1.xml': ['<pageMargins']}

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with printer settings.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_start_page(1)
        worksheet.set_paper(9)
        worksheet.vertical_dpi = 200
        worksheet.write('A1', 'Foo')
        workbook.close()
        self.assertExcelEqual()