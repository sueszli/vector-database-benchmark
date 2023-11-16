from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_filename('simple03.xlsx')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test worksheet selection and activation.'
        workbook = Workbook(self.got_filename)
        worksheet1 = workbook.add_worksheet()
        worksheet2 = workbook.add_worksheet('Data Sheet')
        worksheet3 = workbook.add_worksheet()
        bold = workbook.add_format({'bold': 1})
        worksheet1.write('A1', 'Foo')
        worksheet1.write('A2', 123)
        worksheet3.write('B2', 'Foo')
        worksheet3.write('B3', 'Bar', bold)
        worksheet3.write('C4', 234)
        worksheet2.activate()
        worksheet2.select()
        worksheet3.select()
        worksheet3.activate()
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_in_memory(self):
        if False:
            print('Hello World!')
        'Test worksheet selection and activation.'
        workbook = Workbook(self.got_filename, {'in_memory': True})
        worksheet1 = workbook.add_worksheet()
        worksheet2 = workbook.add_worksheet('Data Sheet')
        worksheet3 = workbook.add_worksheet()
        bold = workbook.add_format({'bold': 1})
        worksheet1.write('A1', 'Foo')
        worksheet1.write('A2', 123)
        worksheet3.write('B2', 'Foo')
        worksheet3.write('B3', 'Bar', bold)
        worksheet3.write('C4', 234)
        worksheet2.activate()
        worksheet2.select()
        worksheet3.select()
        worksheet3.activate()
        workbook.close()
        self.assertExcelEqual()