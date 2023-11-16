from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook
from io import BytesIO

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_filename('header_image04.xlsx')
        self.ignore_elements = {'xl/worksheets/sheet1.xml': ['<pageMargins', '<pageSetup']}

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_footer('&L&G&C&G&R&G', {'image_left': self.image_dir + 'red.jpg', 'image_center': self.image_dir + 'blue.jpg', 'image_right': self.image_dir + 'yellow.jpg'})
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_with_picture(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_footer('&L&[Picture]&C&G&R&[Picture]', {'image_left': self.image_dir + 'red.jpg', 'image_center': self.image_dir + 'blue.jpg', 'image_right': self.image_dir + 'yellow.jpg'})
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_from_bytesio(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        image_file_left = open(self.image_dir + 'red.jpg', 'rb')
        image_data_left = BytesIO(image_file_left.read())
        image_file_left.close()
        image_file_center = open(self.image_dir + 'blue.jpg', 'rb')
        image_data_center = BytesIO(image_file_center.read())
        image_file_center.close()
        image_file_right = open(self.image_dir + 'yellow.jpg', 'rb')
        image_data_right = BytesIO(image_file_right.read())
        image_file_right.close()
        worksheet.set_footer('&L&G&C&G&R&G', {'image_left': 'red.jpg', 'image_center': 'blue.jpg', 'image_right': 'yellow.jpg', 'image_data_left': image_data_left, 'image_data_center': image_data_center, 'image_data_right': image_data_right})
        workbook.close()
        self.assertExcelEqual()