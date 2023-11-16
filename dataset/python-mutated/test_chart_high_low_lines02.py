from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('chart_high_low_lines02.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of an XlsxWriter file with high-low lines.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        chart = workbook.add_chart({'type': 'line'})
        chart.axis_ids = [61180928, 63898368]
        data = [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15]]
        worksheet.write_column('A1', data[0])
        worksheet.write_column('B1', data[1])
        worksheet.write_column('C1', data[2])
        chart.set_high_low_lines({'line': {'color': 'red', 'dash_type': 'square_dot'}})
        chart.add_series({'categories': '=Sheet1!$A$1:$A$5', 'values': '=Sheet1!$B$1:$B$5'})
        chart.add_series({'categories': '=Sheet1!$A$1:$A$5', 'values': '=Sheet1!$C$1:$C$5'})
        worksheet.insert_chart('E9', chart)
        workbook.close()
        self.assertExcelEqual()