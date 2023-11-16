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
        self.set_filename('optimize11.xlsx')

    def test_create_file_no_close(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename, {'constant_memory': True, 'in_memory': False})
        for i in range(1, 10):
            worksheet = workbook.add_worksheet()
            worksheet.write('A1', 'Hello 1')
            worksheet.write('A2', 'Hello 2')
            worksheet.write('A4', 'Hello 3')
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_with_close(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename, {'constant_memory': True, 'in_memory': False})
        for i in range(1, 10):
            worksheet = workbook.add_worksheet()
            worksheet.write('A1', 'Hello 1')
            worksheet.write('A2', 'Hello 2')
            worksheet.write('A4', 'Hello 3')
            worksheet._opt_close()
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_with_reopen(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename, {'constant_memory': True, 'in_memory': False})
        for i in range(1, 10):
            worksheet = workbook.add_worksheet()
            worksheet.write('A1', 'Hello 1')
            worksheet._opt_close()
            worksheet._opt_reopen()
            worksheet.write('A2', 'Hello 2')
            worksheet._opt_close()
            worksheet._opt_reopen()
            worksheet.write('A4', 'Hello 3')
            worksheet._opt_close()
            worksheet._opt_reopen()
            worksheet._opt_close()
        workbook.close()
        self.assertExcelEqual()