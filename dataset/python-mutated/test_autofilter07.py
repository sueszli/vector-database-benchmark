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
        self.set_filename('autofilter07.xlsx')
        self.set_text_file('autofilter_data.txt')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        '\n        Test the creation of a simple XlsxWriter file with an autofilter.\n        Test autofilters where column filter ids are relative to autofilter\n        range.\n        '
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.autofilter('D3:G53')
        worksheet.filter_column('D', 'region == East')
        textfile = open(self.txt_filename)
        headers = textfile.readline().strip('\n').split()
        worksheet.write_row('D3', headers)
        row = 3
        for line in textfile:
            data = line.strip('\n').split()
            for (i, item) in enumerate(data):
                try:
                    data[i] = float(item)
                except ValueError:
                    pass
            region = data[0]
            if region == 'East':
                pass
            else:
                worksheet.set_row(row, options={'hidden': True})
            worksheet.write_row(row, 3, data)
            row += 1
        textfile.close()
        workbook.close()
        self.assertExcelEqual()