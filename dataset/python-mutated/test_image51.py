from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('image51.xlsx')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.insert_image('E9', self.image_dir + 'red.png', {'url': 'https://duckduckgo.com/?q=1'})
        worksheet.insert_image('E13', self.image_dir + 'red2.png', {'url': 'https://duckduckgo.com/?q=2'})
        workbook.close()
        self.assertExcelEqual()