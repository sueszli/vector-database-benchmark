from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_filename('chart_combined01.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        chart1 = workbook.add_chart({'type': 'column'})
        chart1.axis_ids = [84882560, 84884096]
        data = [[2, 7, 3, 6, 2], [20, 25, 10, 10, 20]]
        worksheet.write_column('A1', data[0])
        worksheet.write_column('B1', data[1])
        chart1.add_series({'values': '=Sheet1!$A$1:$A$5'})
        chart1.add_series({'values': '=Sheet1!$B$1:$B$5'})
        worksheet.insert_chart('E9', chart1)
        workbook.close()
        self.assertExcelEqual()