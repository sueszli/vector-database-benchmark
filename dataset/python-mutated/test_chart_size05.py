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
        self.set_filename('chart_size04.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test XlsxWriter chartarea properties.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        chart = workbook.add_chart({'type': 'column'})
        chart.axis_ids = [73773440, 73774976]
        data = [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15]]
        worksheet.write_column('A1', data[0])
        worksheet.write_column('B1', data[1])
        worksheet.write_column('C1', data[2])
        chart.add_series({'values': '=Sheet1!$A$1:$A$5'})
        chart.add_series({'values': '=Sheet1!$B$1:$B$5'})
        chart.add_series({'values': '=Sheet1!$C$1:$C$5'})
        worksheet.insert_chart('E9', chart, {'x_offset': 8, 'y_offset': 9})
        workbook.close()
        self.assertExcelEqual()