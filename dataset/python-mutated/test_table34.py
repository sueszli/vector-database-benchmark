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
        self.set_filename('table34.xlsx')
        self.ignore_files = ['xl/calcChain.xml', '[Content_Types].xml', 'xl/_rels/workbook.xml.rels']

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with tables.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        format1 = workbook.add_format({'num_format': '0.0000'})
        data = [['Foo', 1234, 0, 4321], ['Bar', 1256, 0, 4320], ['Baz', 2234, 0, 4332], ['Bop', 1324, 0, 4333]]
        worksheet.set_column('C:F', 10.288)
        worksheet.add_table('C2:F6', {'data': data, 'columns': [{}, {}, {}, {'formula': 'Table1[[#This Row],[Column3]]', 'format': format1}]})
        workbook.close()
        self.assertExcelEqual()