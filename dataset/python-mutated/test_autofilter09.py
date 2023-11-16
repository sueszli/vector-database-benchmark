from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('autofilter09.xlsx')
        self.set_text_file('autofilter_data.txt')

    def test_create_file(self):
        if False:
            print('Hello World!')
        '\n        Test the creation of a simple XlsxWriter file with an autofilter.\n        This test checks a filter list.\n        '
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.autofilter('A1:D51')
        worksheet.filter_column_list(0, ['East', 'South', 'North'])
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
            if region == 'North' or region == 'South' or region == 'East':
                pass
            else:
                worksheet.set_row(row, options={'hidden': True})
            worksheet.write_row(row, 0, data)
            row += 1
        textfile.close()
        workbook.close()
        self.assertExcelEqual()