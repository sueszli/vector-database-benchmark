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
        self.set_filename('hyperlink34.xlsx')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.insert_image('A1', self.image_dir + 'blue.png')
        worksheet.insert_image('B3', self.image_dir + 'red.jpg', {'url': 'https://github.com/jmcnamara'})
        worksheet.insert_image('D5', self.image_dir + 'yellow.jpg')
        worksheet.insert_image('F9', self.image_dir + 'grey.png', {'url': 'https://github.com'})
        workbook.close()
        self.assertExcelEqual()