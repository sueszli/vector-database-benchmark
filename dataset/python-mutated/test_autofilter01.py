from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('autofilter01.xlsx')
        self.set_text_file('autofilter_data.txt')

    def test_create_file(self):
        if False:
            return 10
        '\n        Test the creation of a simple XlsxWriter file with an autofilter.\n        This test corresponds to the following examples/autofilter.py example:\n        Example 1. Autofilter without conditions.\n        '
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.autofilter('A1:D51')
        textfile = open(self.txt_filename)
        row = 0
        for line in textfile:
            data = line.strip('\n').split()
            for (i, item) in enumerate(data):
                try:
                    data[i] = float(item)
                except ValueError:
                    pass
            worksheet.write_row(row, 0, data)
            row += 1
        textfile.close()
        workbook.close()
        self.assertExcelEqual()