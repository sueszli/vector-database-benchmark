from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('hyperlink09.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with hyperlinks.'
        workbook = Workbook(self.got_filename)
        workbook.default_url_format = None
        worksheet = workbook.add_worksheet()
        worksheet.write_url('A1', 'external:..\\foo.xlsx')
        worksheet.write_url('A3', 'external:..\\foo.xlsx#Sheet1!A1')
        worksheet.write_url('A5', 'external:\\\\VBOXSVR\\share\\foo.xlsx#Sheet1!B2', None, 'J:\\foo.xlsx#Sheet1!B2')
        workbook.close()
        self.assertExcelEqual()