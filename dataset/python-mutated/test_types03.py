from ..excel_comparison_test import ExcelComparisonTest
from decimal import Decimal
from fractions import Fraction
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('types03.xlsx')

    def test_write_number_float(self):
        if False:
            i = 10
            return i + 15
        'Test writing number types.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 0.5)
        worksheet.write_number('A2', 0.5)
        workbook.close()
        self.assertExcelEqual()

    def test_write_number_decimal(self):
        if False:
            i = 10
            return i + 15
        'Test writing number types.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', Decimal('0.5'))
        worksheet.write_number('A2', Decimal('0.5'))
        workbook.close()
        self.assertExcelEqual()

    def test_write_number_fraction(self):
        if False:
            print('Hello World!')
        'Test writing number types.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', Fraction(1, 2))
        worksheet.write_number('A2', Fraction(2, 4))
        workbook.close()
        self.assertExcelEqual()