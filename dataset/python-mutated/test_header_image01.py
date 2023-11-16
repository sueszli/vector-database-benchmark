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
        self.set_filename('header_image01.xlsx')
        self.ignore_elements = {'xl/worksheets/sheet1.xml': ['<pageMargins', '<pageSetup']}

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_header('&L&G', {'image_left': self.image_dir + 'red.jpg'})
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_in_memory(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename, {'in_memory': True})
        worksheet = workbook.add_worksheet()
        worksheet.set_header('&L&G', {'image_left': self.image_dir + 'red.jpg'})
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_from_bytesio(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        image_file = open(self.image_dir + 'red.jpg', 'rb')
        image_data = BytesIO(image_file.read())
        image_file.close()
        worksheet.set_header('&L&G', {'image_left': 'red.jpg', 'image_data_left': image_data})
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_from_bytesio_in_memory(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename, {'in_memory': True})
        worksheet = workbook.add_worksheet()
        image_file = open(self.image_dir + 'red.jpg', 'rb')
        image_data = BytesIO(image_file.read())
        image_file.close()
        worksheet.set_header('&L&G', {'image_left': 'red.jpg', 'image_data_left': image_data})
        workbook.close()
        self.assertExcelEqual()