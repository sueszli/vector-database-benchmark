from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('image_anchor03.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.insert_image('E9', self.image_dir + 'red.png', {'positioning': 1})
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_in_memory(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename, {'in_memory': True})
        worksheet = workbook.add_worksheet()
        worksheet.insert_image('E9', self.image_dir + 'red.png', {'positioning': 1})
        workbook.close()
        self.assertExcelEqual()