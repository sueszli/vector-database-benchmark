from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('textbox03.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file with textbox(s).'
        workbook = Workbook(self.got_filename)
        worksheet1 = workbook.add_worksheet()
        worksheet2 = workbook.add_worksheet()
        worksheet1.insert_textbox('E9', 'This is some text')
        worksheet1.insert_textbox('H18', 'Some more text')
        worksheet2.insert_textbox('C4', 'Hello')
        workbook.close()
        self.assertExcelEqual()