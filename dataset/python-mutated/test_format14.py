from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('format14.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the center across format.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        center = workbook.add_format()
        center.set_center_across()
        worksheet.write('A1', 'foo', center)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_2(self):
        if False:
            return 10
        'Test the center across format.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        center = workbook.add_format({'center_across': True})
        worksheet.write('A1', 'foo', center)
        workbook.close()
        self.assertExcelEqual()