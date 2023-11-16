from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_filename('macro03.xlsm')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet('Foo')
        workbook.add_vba_project(self.vba_dir + 'vbaProject04.bin')
        worksheet.write('A1', 123)
        workbook.close()
        self.assertExcelEqual()