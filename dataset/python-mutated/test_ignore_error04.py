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
        self.set_filename('ignore_error04.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_string('A1', '123')
        worksheet.write_string('C3', '123')
        worksheet.write_string('E5', '123')
        worksheet.ignore_errors({'number_stored_as_text': 'A1 C3 E5'})
        workbook.close()
        self.assertExcelEqual()