from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_filename('tutorial02.xlsx')
        self.ignore_files = ['xl/calcChain.xml', '[Content_Types].xml', 'xl/_rels/workbook.xml.rels']

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Example spreadsheet used in the tutorial 2.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        bold = workbook.add_format({'bold': True})
        money_format = workbook.add_format({'num_format': '\\$#,##0'})
        worksheet.write('A1', 'Item', bold)
        worksheet.write('B1', 'Cost', bold)
        expenses = (['Rent', 1000], ['Gas', 100], ['Food', 300], ['Gym', 50])
        row = 1
        col = 0
        for (item, cost) in expenses:
            worksheet.write(row, col, item)
            worksheet.write(row, col + 1, cost, money_format)
            row += 1
        worksheet.write(row, 0, 'Total', bold)
        worksheet.write(row, 1, '=SUM(B2:B5)', money_format, 1450)
        workbook.close()
        self.assertExcelEqual()