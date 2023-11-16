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
        self.set_filename('dynamic_array02.xlsx')

    def test_dynamic_array02_1(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_dynamic_array_formula('B1:B1', '=_xlfn.UNIQUE(A1)', None, 0)
        worksheet.write('A1', 0)
        workbook.close()
        self.assertExcelEqual()

    def test_dynamic_array02_2(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_dynamic_array_formula('B1', '=_xlfn.UNIQUE(A1)', None, 0)
        worksheet.write('A1', 0)
        workbook.close()
        self.assertExcelEqual()

    def test_dynamic_array02_3(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write('B1', '=_xlfn.UNIQUE(A1)', None, 0)
        worksheet.write('A1', 0)
        workbook.close()
        self.assertExcelEqual()

    def test_dynamic_array02_4(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_dynamic_array_formula('B1', '=UNIQUE(A1)', None, 0)
        worksheet.write('A1', 0)
        workbook.close()
        self.assertExcelEqual()

    def test_dynamic_array02_5(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_formula('B1', '=UNIQUE(A1)', None, 0)
        worksheet.write('A1', 0)
        workbook.close()
        self.assertExcelEqual()

    def test_dynamic_array02_6(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write('B1', '=UNIQUE(A1)', None, 0)
        worksheet.write('A1', 0)
        workbook.close()
        self.assertExcelEqual()

    def test_dynamic_array02_7(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_array_formula('B1', '=UNIQUE(A1)', None, 0)
        worksheet.write('A1', 0)
        workbook.close()
        self.assertExcelEqual()