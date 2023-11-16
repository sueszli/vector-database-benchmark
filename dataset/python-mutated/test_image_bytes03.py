from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook
from io import BytesIO

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('image03.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        image_file = open(self.image_dir + 'red.jpg', 'rb')
        image_data = BytesIO(image_file.read())
        image_file.close()
        worksheet.insert_image('E9', 'red.jpg', {'image_data': image_data})
        workbook.close()
        self.assertExcelEqual()