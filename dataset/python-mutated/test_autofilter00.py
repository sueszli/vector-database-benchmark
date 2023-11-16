from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('autofilter00.xlsx')
        self.set_text_file('autofilter_data.txt')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the creation of a simple XlsxWriter file with an autofilter.\n        This test is the base comparison. It has data but no autofilter.\n        '
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        textfile = open(self.txt_filename)
        row = 0
        for line in textfile:
            data = line.strip('\n').split()
            for (i, item) in enumerate(data):
                try:
                    data[i] = float(item)
                except ValueError:
                    pass
            for col in range(len(data)):
                worksheet.write(row, col, data[col])
            row += 1
        textfile.close()
        workbook.close()
        self.assertExcelEqual()