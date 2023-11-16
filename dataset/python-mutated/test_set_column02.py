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
        self.set_filename('set_column01.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column('A:A', 0.083333333333333)
        worksheet.set_column('B:B', 0.166666666666667)
        worksheet.set_column('C:C', 0.25)
        worksheet.set_column('D:D', 0.333333333333333)
        worksheet.set_column('E:E', 0.416666666666667)
        worksheet.set_column('F:F', 0.5)
        worksheet.set_column('G:G', 0.583333333333333)
        worksheet.set_column('H:H', 0.666666666666666)
        worksheet.set_column('I:I', 0.75)
        worksheet.set_column('J:J', 0.833333333333333)
        worksheet.set_column('K:K', 0.916666666666666)
        worksheet.set_column('L:L', 1.0)
        worksheet.set_column('M:M', 1.14285714285714)
        worksheet.set_column('N:N', 1.28571428571429)
        worksheet.set_column('O:O', 1.42857142857143)
        worksheet.set_column('P:P', 1.57142857142857)
        worksheet.set_column('Q:Q', 1.71428571428571)
        worksheet.set_column('R:R', 1.85714285714286)
        worksheet.set_column('S:S', 2.0)
        worksheet.set_column('T:T', 2.14285714285714)
        worksheet.set_column('U:U', 2.28571428571429)
        worksheet.set_column('V:V', 2.42857142857143)
        worksheet.set_column('W:W', 2.57142857142857)
        worksheet.set_column('X:X', 2.71428571428571)
        worksheet.set_column('Y:Y', 2.85714285714286)
        worksheet.set_column('Z:Z', 3.0)
        worksheet.set_column('AB:AB', 8.57142857142857)
        worksheet.set_column('AC:AC', 8.71142857142857)
        worksheet.set_column('AD:AD', 8.85714285714286)
        worksheet.set_column('AE:AE', 9.0)
        worksheet.set_column('AF:AF', 9.14285714285714)
        worksheet.set_column('AG:AG', 9.28571428571429)
        workbook.close()
        self.assertExcelEqual()