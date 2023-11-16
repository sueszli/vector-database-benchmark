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
        self.set_filename('image26.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.insert_image('B2', self.image_dir + 'black_72.png')
        worksheet.insert_image('B8', self.image_dir + 'black_96.png')
        worksheet.insert_image('B13', self.image_dir + 'black_150.png')
        worksheet.insert_image('B17', self.image_dir + 'black_300.png')
        workbook.close()
        self.assertExcelEqual()