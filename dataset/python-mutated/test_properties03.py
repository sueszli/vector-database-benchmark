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
        self.set_filename('properties03.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        workbook.set_custom_property('Checked by', 'Adam')
        worksheet.set_column('A:A', 70)
        worksheet.write('A1', "Select 'Office Button -> Prepare -> Properties' to see the file properties.")
        workbook.close()
        self.assertExcelEqual()