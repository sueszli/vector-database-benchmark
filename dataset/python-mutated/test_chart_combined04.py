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
        self.set_filename('chart_combined04.xlsx')
        self.ignore_elements = {'xl/charts/chart1.xml': ['<c:dispBlanksAs', '<c:tickLblPos', '<c:crosses']}

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        chart1 = workbook.add_chart({'type': 'column'})
        chart2 = workbook.add_chart({'type': 'line'})
        data = [[2, 7, 3, 6, 2], [20, 25, 10, 10, 20]]
        worksheet.write_column('A1', data[0])
        worksheet.write_column('B1', data[1])
        chart1.add_series({'values': '=Sheet1!$A$1:$A$5'})
        chart2.add_series({'values': '=Sheet1!$B$1:$B$5', 'y2_axis': 1})
        chart1.combine(chart2)
        worksheet.insert_chart('E9', chart1)
        workbook.close()
        self.assertExcelEqual()