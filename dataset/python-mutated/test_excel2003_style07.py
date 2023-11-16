from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('excel2003_style07.xlsx')
        self.ignore_elements = {'xl/drawings/drawing1.xml': ['<xdr:cNvPr', '<a:picLocks', '<a:srcRect/>', '<xdr:spPr', '<a:noFill/>']}

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename, {'excel2003_style': True})
        worksheet = workbook.add_worksheet()
        worksheet.insert_image('B3', self.image_dir + 'yellow.jpg', {'x_offset': 4, 'y_offset': 3})
        workbook.close()
        self.assertExcelEqual()