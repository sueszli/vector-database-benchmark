from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('dynamic_array01.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_dynamic_array_formula('A1:A1', '=AVERAGE(TIMEVALUE(B1:B2))', None, 0)
        worksheet.write('B1', '12:00')
        worksheet.write('B2', '12:00')
        workbook.close()
        self.assertExcelEqual()