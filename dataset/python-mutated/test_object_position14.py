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
        self.set_filename('object_position14.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column(1, 1, 5, None, {'hidden': 1})
        worksheet.insert_image('E9', self.image_dir + 'red.png')
        workbook.close()
        self.assertExcelEqual()