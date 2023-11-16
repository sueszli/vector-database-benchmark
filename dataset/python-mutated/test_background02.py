from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook
from io import BytesIO

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('background02.xlsx')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of an XlsxWriter file with a background image.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_background(self.image_dir + 'logo.jpg')
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_bytestream(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of an XlsxWriter file with a background image.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        image_file = open(self.image_dir + 'logo.jpg', 'rb')
        image_data = BytesIO(image_file.read())
        image_file.close()
        worksheet.set_background(image_data, is_byte_stream=True)
        workbook.close()
        self.assertExcelEqual()