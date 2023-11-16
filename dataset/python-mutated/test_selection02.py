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
        self.set_filename('selection02.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet1 = workbook.add_worksheet()
        worksheet2 = workbook.add_worksheet()
        worksheet3 = workbook.add_worksheet()
        worksheet4 = workbook.add_worksheet()
        worksheet5 = workbook.add_worksheet()
        worksheet6 = workbook.add_worksheet()
        worksheet1.set_selection(3, 2, 3, 2)
        worksheet2.set_selection(3, 2, 6, 6)
        worksheet3.set_selection(6, 6, 3, 2)
        worksheet4.set_selection('C4')
        worksheet5.set_selection('C4:G7')
        worksheet6.set_selection('G7:C4')
        workbook.close()
        self.assertExcelEqual()