from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('data_validation02.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a XlsxWriter file with data validation.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.data_validation('C2', {'validate': 'list', 'value': ['Foo', 'Bar', 'Baz'], 'input_title': 'This is the input title', 'input_message': 'This is the input message'})
        workbook.close()
        self.assertExcelEqual()