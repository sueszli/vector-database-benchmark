from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('background01.xlsx')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of an XlsxWriter file with a background image.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.insert_image('E9', self.image_dir + 'logo.jpg')
        workbook.close()
        self.assertExcelEqual()