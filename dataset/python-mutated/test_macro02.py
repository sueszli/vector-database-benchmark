from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('macro02.xlsm')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        workbook.add_vba_project(self.vba_dir + 'vbaProject03.bin')
        workbook.set_vba_name('MyWorkbook')
        worksheet.set_vba_name('MySheet1')
        worksheet.write('A1', 123)
        workbook.close()
        self.assertExcelEqual()