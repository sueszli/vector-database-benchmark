from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('table32.xlsx')
        self.ignore_files = ['xl/calcChain.xml', '[Content_Types].xml', 'xl/_rels/workbook.xml.rels']

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with tables.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column('C:F', 10.288)
        worksheet.write_string('A1', 'Column1')
        worksheet.write_string('B1', 'Column2')
        worksheet.write_string('C1', 'Column3')
        worksheet.write_string('D1', 'Column4')
        worksheet.write_string('E1', 'Total')
        worksheet.add_table('C3:F14', {'total_row': 1, 'columns': [{'total_string': 'Total'}, {'total_function': 'D5+D9'}, {'total_function': '=SUM([Column3])'}, {'total_function': 'count'}]})
        workbook.close()
        self.assertExcelEqual()