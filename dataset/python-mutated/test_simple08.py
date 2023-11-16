from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('simple08.xlsx')

    def test_create_file(self):
        if False:
            return 10
        "Test '0' number format. GH103."
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        format1 = workbook.add_format({'num_format': 1})
        worksheet.write(0, 0, 1.23, format1)
        workbook.close()
        self.assertExcelEqual()