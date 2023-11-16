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
        self.set_filename('simple07.xlsx')

    def test_write_nan(self):
        if False:
            for i in range(10):
                print('nop')
        'Test write with NAN/INF. Issue #30'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_string(0, 0, 'Foo')
        worksheet.write_number(1, 0, 123)
        worksheet.write_string(2, 0, 'NAN')
        worksheet.write_string(3, 0, 'nan')
        worksheet.write_string(4, 0, 'INF')
        worksheet.write_string(5, 0, 'infinity')
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_in_memory(self):
        if False:
            print('Hello World!')
        'Test write with NAN/INF. Issue #30'
        workbook = Workbook(self.got_filename, {'in_memory': True})
        worksheet = workbook.add_worksheet()
        worksheet.write_string(0, 0, 'Foo')
        worksheet.write_number(1, 0, 123)
        worksheet.write_string(2, 0, 'NAN')
        worksheet.write_string(3, 0, 'nan')
        worksheet.write_string(4, 0, 'INF')
        worksheet.write_string(5, 0, 'infinity')
        workbook.close()
        self.assertExcelEqual()