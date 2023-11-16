from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('optimize02.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename, {'constant_memory': True, 'in_memory': False})
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Hello')
        worksheet.write('A2', 123)
        worksheet.write('G1', 'Foo')
        workbook.close()
        self.assertExcelEqual()