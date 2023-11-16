from ..excel_comparison_test import ExcelComparisonTest
from datetime import date
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_filename('date_1904_01.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a XlsxWriter file with date times in 1900 and\n        1904 epochs.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        format1 = workbook.add_format({'num_format': 14})
        worksheet.set_column('A:A', 12)
        worksheet.write_datetime('A1', date(1900, 1, 1), format1)
        worksheet.write_datetime('A2', date(1902, 9, 26), format1)
        worksheet.write_datetime('A3', date(1913, 9, 8), format1)
        worksheet.write_datetime('A4', date(1927, 5, 18), format1)
        worksheet.write_datetime('A5', date(2173, 10, 14), format1)
        worksheet.write_datetime('A6', date(4637, 11, 26), format1)
        workbook.close()
        self.assertExcelEqual()