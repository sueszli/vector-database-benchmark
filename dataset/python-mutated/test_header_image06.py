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
        self.set_filename('header_image06.xlsx')
        self.ignore_elements = {'xl/worksheets/sheet1.xml': ['<pageMargins', '<pageSetup'], 'xl/worksheets/sheet2.xml': ['<pageMargins', '<pageSetup']}

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet1 = workbook.add_worksheet()
        worksheet2 = workbook.add_worksheet()
        worksheet1.set_header('&L&G', {'image_left': self.image_dir + 'red.jpg'})
        worksheet2.set_header('&L&G', {'image_left': self.image_dir + 'blue.jpg'})
        workbook.close()
        self.assertExcelEqual()