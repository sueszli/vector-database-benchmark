from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('tutorial01.xlsx')
        self.ignore_files = ['xl/calcChain.xml', '[Content_Types].xml', 'xl/_rels/workbook.xml.rels']

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Example spreadsheet used in the tutorial.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        expenses = (['Rent', 1000], ['Gas', 100], ['Food', 300], ['Gym', 50])
        row = 0
        col = 0
        for (item, cost) in expenses:
            worksheet.write(row, col, item)
            worksheet.write(row, col + 1, cost)
            row += 1
        worksheet.write(row, 0, 'Total')
        worksheet.write(row, 1, '=SUM(B1:B4)', None, 1450)
        workbook.close()
        self.assertExcelEqual()