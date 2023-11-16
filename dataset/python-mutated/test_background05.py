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
        self.set_filename('background05.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of an XlsxWriter file with a background image.'
        workbook = Workbook(self.got_filename)
        worksheet1 = workbook.add_worksheet()
        worksheet2 = workbook.add_worksheet()
        worksheet1.set_background(self.image_dir + 'logo.jpg')
        worksheet2.set_background(self.image_dir + 'red.jpg')
        workbook.close()
        self.assertExcelEqual()