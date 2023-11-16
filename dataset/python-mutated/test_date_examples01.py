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
        self.set_filename('date_examples01.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Example spreadsheet used in the tutorial.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column('A:A', 30)
        number = 41333.5
        worksheet.write('A1', number)
        format2 = workbook.add_format({'num_format': 'dd/mm/yy'})
        worksheet.write('A2', number, format2)
        format3 = workbook.add_format({'num_format': 'mm/dd/yy'})
        worksheet.write('A3', number, format3)
        format4 = workbook.add_format({'num_format': 'd\\-m\\-yyyy'})
        worksheet.write('A4', number, format4)
        format5 = workbook.add_format({'num_format': 'dd/mm/yy\\ hh:mm'})
        worksheet.write('A5', number, format5)
        format6 = workbook.add_format({'num_format': 'd\\ mmm\\ yyyy'})
        worksheet.write('A6', number, format6)
        format7 = workbook.add_format({'num_format': 'mmm\\ d\\ yyyy\\ hh:mm\\ AM/PM'})
        worksheet.write('A7', number, format7)
        workbook.close()
        self.assertExcelEqual()