from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('header_image15.xlsx')
        self.ignore_elements = {'xl/worksheets/sheet1.xml': ['<pageMargins', '<pageSetup'], 'xl/worksheets/sheet2.xml': ['<pageMargins', '<pageSetup']}

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with image(s).'
        workbook = Workbook(self.got_filename)
        worksheet1 = workbook.add_worksheet()
        worksheet2 = workbook.add_worksheet()
        worksheet1.set_header('&L&G', {'image_left': self.image_dir + 'red.jpg'})
        worksheet2.set_header('&L&G', {'image_left': self.image_dir + 'red.jpg'})
        workbook.close()
        self.assertExcelEqual()