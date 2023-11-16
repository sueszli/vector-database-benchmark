from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('button07.xlsm')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        workbook.set_vba_name()
        worksheet.set_vba_name()
        worksheet.insert_button('C2', {'macro': 'say_hello', 'caption': 'Hello'})
        workbook.add_vba_project(self.vba_dir + 'vbaProject02.bin')
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_explicit_vba_names(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        workbook.set_vba_name('ThisWorkbook')
        worksheet.set_vba_name('Sheet1')
        worksheet.insert_button('C2', {'macro': 'say_hello', 'caption': 'Hello'})
        workbook.add_vba_project(self.vba_dir + 'vbaProject02.bin')
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_implicit_vba_names(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.insert_button('C2', {'macro': 'say_hello', 'caption': 'Hello'})
        workbook.add_vba_project(self.vba_dir + 'vbaProject02.bin')
        workbook.close()
        self.assertExcelEqual()