from ..excel_comparison_test import ExcelComparisonTest
from datetime import date
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_filename('simple01.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple workbook.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_string(0, 0, 'Hello')
        worksheet.write_number(1, 0, 123)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_A1(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple workbook with A1 notation.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_string('A1', 'Hello')
        worksheet.write_number('A2', 123)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_write(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple workbook using write().'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write(0, 0, 'Hello')
        worksheet.write(1, 0, 123)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_with_statement(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple workbook using `with` statement.'
        with Workbook(self.got_filename) as workbook:
            worksheet = workbook.add_worksheet()
            worksheet.write(0, 0, 'Hello')
            worksheet.write(1, 0, 123)
        self.assertExcelEqual()

    def test_create_file_write_A1(self):
        if False:
            return 10
        'Test the creation of a simple workbook using write() with A1.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Hello')
        worksheet.write('A2', 123)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_kwargs(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple workbook with keyword args.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_string(row=0, col=0, string='Hello')
        worksheet.write_number(row=1, col=0, number=123)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_write_date_default(self):
        if False:
            i = 10
            return i + 15
        'Test writing a datetime without a format. Issue #33'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Hello')
        worksheet.write('A2', date(1900, 5, 2))
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_in_memory(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple workbook.'
        workbook = Workbook(self.got_filename, {'in_memory': True})
        worksheet = workbook.add_worksheet()
        worksheet.write_string(0, 0, 'Hello')
        worksheet.write_number(1, 0, 123)
        workbook.close()
        self.assertExcelEqual()