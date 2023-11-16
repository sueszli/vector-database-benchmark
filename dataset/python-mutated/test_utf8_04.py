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
        self.set_filename('utf8_04.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of an XlsxWriter file with utf-8 strings.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet('Café & Café')
        worksheet.write('A1', 'Café & Café')
        workbook.close()
        self.assertExcelEqual()