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
        self.set_filename('chart_column04.xlsx')
        self.ignore_elements = {'xl/workbook.xml': ['<fileVersion', '<calcPr']}

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        chart = workbook.add_chart({'type': 'column'})
        chart.axis_ids = [63591936, 63593856]
        chart.axis2_ids = [63613568, 63612032]
        data = [[1, 2, 3, 4, 5], [6, 8, 6, 4, 2]]
        worksheet.write_column('A1', data[0])
        worksheet.write_column('B1', data[1])
        chart.add_series({'values': '=Sheet1!$A$1:$A$5'})
        chart.add_series({'values': '=Sheet1!$B$1:$B$5', 'y2_axis': 1})
        worksheet.insert_chart('E9', chart)
        workbook.close()
        self.assertExcelEqual()