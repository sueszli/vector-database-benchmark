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
        self.set_filename('image33.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column('D:D', 3.86)
        worksheet.set_column('E:E', 1.43)
        worksheet.set_row(7, 7.5)
        worksheet.set_row(8, 9.75)
        worksheet.insert_image('E9', self.image_dir + 'red.png', {'x_offset': -2, 'y_offset': -1})
        workbook.close()
        self.assertExcelEqual()