from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('format11.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test a vertical and horizontal centered format.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        centered = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
        worksheet.write('B2', 'Foo', centered)
        workbook.close()
        self.assertExcelEqual()