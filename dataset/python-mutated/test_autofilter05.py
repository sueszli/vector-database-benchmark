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
        self.set_filename('autofilter05.xlsx')
        self.set_text_file('autofilter_data.txt')

    def test_create_file(self):
        if False:
            print('Hello World!')
        '\n        Test the creation of a simple XlsxWriter file with an autofilter.\n        This test corresponds to the following examples/autofilter.pl example:\n        Example 5. Autofilter with filter for blanks.\n        '
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.autofilter('A1:D51')
        worksheet.filter_column(0, 'x == Blanks')
        textfile = open(self.txt_filename)
        headers = textfile.readline().strip('\n').split()
        worksheet.write_row('A1', headers)
        row = 1
        for line in textfile:
            data = line.strip('\n').split()
            for (i, item) in enumerate(data):
                try:
                    data[i] = float(item)
                except ValueError:
                    pass
            if row == 6:
                data[0] = ''
            region = data[0]
            if region == '':
                pass
            else:
                worksheet.set_row(row, options={'hidden': True})
            worksheet.write_row(row, 0, data)
            row += 1
        textfile.close()
        workbook.close()
        self.assertExcelEqual()