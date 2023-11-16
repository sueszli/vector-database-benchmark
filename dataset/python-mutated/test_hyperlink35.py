from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('hyperlink35.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.insert_image('A1', self.image_dir + 'blue.png', {'url': 'https://github.com/foo'})
        worksheet.insert_image('B3', self.image_dir + 'red.jpg', {'url': 'https://github.com/bar'})
        worksheet.insert_image('D5', self.image_dir + 'yellow.jpg', {'url': 'https://github.com/baz'})
        worksheet.insert_image('F9', self.image_dir + 'grey.png', {'url': 'https://github.com/boo'})
        workbook.close()
        self.assertExcelEqual()