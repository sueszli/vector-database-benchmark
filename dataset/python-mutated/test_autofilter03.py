from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('autofilter03.xlsx')
        self.set_text_file('autofilter_data.txt')

    def test_create_file(self):
        if False:
            print('Hello World!')
        '\n        Test the creation of a simple XlsxWriter file with an autofilter.\n        This test corresponds to the following examples/autofilter.py example:\n        Example 3. Autofilter with a dual filter condition in one of the\n        columns.\n        '
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.autofilter('A1:D51')
        worksheet.filter_column(0, 'x == East or x = South')
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
            region = data[0]
            if region in ('East', 'South'):
                pass
            else:
                worksheet.set_row(row, options={'hidden': True})
            worksheet.write_row(row, 0, data)
            row += 1
        textfile.close()
        workbook.close()
        self.assertExcelEqual()