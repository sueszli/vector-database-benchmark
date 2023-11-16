from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_filename('image18.xlsx')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_row(1, 96)
        worksheet.set_column('C:C', 18)
        worksheet.insert_image('C2', self.image_dir + 'issue32.png', {'x_offset': 5, 'y_offset': 5})
        workbook.close()
        self.assertExcelEqual()