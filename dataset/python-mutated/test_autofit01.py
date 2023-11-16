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
        self.set_filename('autofit01.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.autofit()
        worksheet.write_string(0, 0, 'A')
        worksheet.set_column('A:A', None)
        worksheet.autofit()
        worksheet.set_column('A:A', 0)
        worksheet.autofit()
        worksheet.set_column('A:A', 1.57143)
        worksheet.autofit()
        workbook.close()
        self.assertExcelEqual()