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
        self.set_filename('format01.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with unused formats.'
        workbook = Workbook(self.got_filename)
        worksheet1 = workbook.add_worksheet()
        worksheet2 = workbook.add_worksheet('Data Sheet')
        worksheet3 = workbook.add_worksheet()
        unused1 = workbook.add_format({'bold': 1})
        bold = workbook.add_format({'bold': 1})
        unused2 = workbook.add_format({'bold': 1})
        unused3 = workbook.add_format({'italic': 1})
        worksheet1.write('A1', 'Foo')
        worksheet1.write('A2', 123)
        worksheet3.write('B2', 'Foo')
        worksheet3.write('B3', 'Bar', bold)
        worksheet3.write('C4', 234)
        workbook.close()
        self.assertExcelEqual()