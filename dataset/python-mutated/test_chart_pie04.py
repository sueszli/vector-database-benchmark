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
        self.set_filename('chart_pie04.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        chart = workbook.add_chart({'type': 'pie'})
        data = [[2, 4, 6], [60, 30, 10]]
        worksheet.write_column('A1', data[0])
        worksheet.write_column('B1', data[1])
        chart.add_series({'categories': '=Sheet1!$A$1:$A$3', 'values': '=Sheet1!$B$1:$B$3'})
        chart.set_legend({'position': 'overlay_right'})
        worksheet.insert_chart('E9', chart)
        workbook.close()
        self.assertExcelEqual()