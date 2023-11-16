from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_filename('ignore_error03.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        for row in range(0, 9 + 1):
            worksheet.write_string(row, 0, '123')
        worksheet.ignore_errors({'number_stored_as_text': 'A1:A10'})
        workbook.close()
        self.assertExcelEqual()