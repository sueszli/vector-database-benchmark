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
        self.set_filename('textbox26.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with textbox(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.insert_textbox('E9', 'This is some text', {'font': {'name': 'Arial', 'size': 11.5, 'pitch_family': 34, 'charset': 0}})
        workbook.close()
        self.assertExcelEqual()