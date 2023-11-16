from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('format23.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file with automatic color.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        format1 = workbook.add_format({'color': 'automatic', 'diag_border': 1, 'diag_type': 2, 'diag_color': 'automatic'})
        worksheet.write(0, 0, 'Foo', format1)
        workbook.close()
        self.assertExcelEqual()