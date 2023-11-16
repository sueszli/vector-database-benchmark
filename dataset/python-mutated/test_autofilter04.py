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
        self.set_filename('autofilter04.xlsx')
        self.set_text_file('autofilter_data.txt')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        '\n        Test the creation of a simple XlsxWriter file with an autofilter.\n        This test corresponds to the following examples/autofilter.py example:\n        Example 4. Autofilter with filter conditions in two columns.\n        '
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.autofilter('A1:D51')
        worksheet.filter_column('A', 'x == East')
        worksheet.filter_column('C', 'x > 3000 and x < 8000')
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
            volume = int(data[2])
            if region == 'East' and volume > 3000 and (volume < 8000):
                pass
            else:
                worksheet.set_row(row, options={'hidden': True})
            worksheet.write_row(row, 0, data)
            row += 1
        textfile.close()
        workbook.close()
        self.assertExcelEqual()