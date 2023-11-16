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
        self.set_filename('background06.xlsx')
        self.ignore_elements = {'xl/worksheets/sheet1.xml': ['<pageSetup']}

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of an XlsxWriter file with a background image.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.insert_image('E9', self.image_dir + 'logo.jpg')
        worksheet.set_background(self.image_dir + 'logo.jpg')
        worksheet.set_header('&C&G', {'image_center': self.image_dir + 'blue.jpg'})
        workbook.close()
        self.assertExcelEqual()