from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('textbox31.xlsx')
        self.ignore_elements = {'xl/drawings/drawing1.xml': ['<a:pPr/>']}

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with textbox(s).'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'This is some text')
        worksheet.insert_textbox('E9', 'This is some text', {'textlink': '=$A$1', 'font': {'bold': True}})
        workbook.close()
        self.assertExcelEqual()