from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('merge_range03.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        format = workbook.add_format({'align': 'center'})
        worksheet.merge_range(1, 1, 1, 2, 'Foo', format)
        worksheet.merge_range(1, 3, 1, 4, 'Foo', format)
        worksheet.merge_range(1, 5, 1, 6, 'Foo', format)
        workbook.close()
        self.assertExcelEqual()