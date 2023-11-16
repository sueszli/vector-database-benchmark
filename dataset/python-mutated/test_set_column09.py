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
        self.set_filename('set_column09.xlsx')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column('A:A', 100)
        worksheet.set_column('F:H', 8)
        worksheet.set_column('C:D', 12)
        worksheet.set_column('A:A', 10)
        worksheet.set_column('XFD:XFD', 5)
        worksheet.set_column('ZZ:ZZ', 3)
        workbook.close()
        self.assertExcelEqual()