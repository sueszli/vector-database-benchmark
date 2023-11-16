from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('chartsheet05.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the worksheet properties of an XlsxWriter chartsheet file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        chartsheet = workbook.add_chartsheet()
        chart = workbook.add_chart({'type': 'bar'})
        chart.axis_ids = [43695104, 43787008]
        data = [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15]]
        worksheet.write_column('A1', data[0])
        worksheet.write_column('B1', data[1])
        worksheet.write_column('C1', data[2])
        chart.add_series({'values': '=Sheet1!$A$1:$A$5'})
        chart.add_series({'values': '=Sheet1!$B$1:$B$5'})
        chart.add_series({'values': '=Sheet1!$C$1:$C$5'})
        chartsheet.set_zoom(75)
        chartsheet.set_chart(chart)
        workbook.close()
        self.assertExcelEqual()