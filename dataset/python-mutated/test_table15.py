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
        self.set_filename('table15.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with tables.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        data = [['Foo', 1234, 2000, 4321], ['Bar', 1256, 0, 4320], ['Baz', 2234, 3000, 4332], ['Bop', 1324, 1000, 4333]]
        worksheet.set_column('C:F', 10.288)
        worksheet.add_table('C2:F6', {'data': data})
        workbook.close()
        self.assertExcelEqual()