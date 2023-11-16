from ..excel_comparison_test import ExcelComparisonTest
from datetime import datetime
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_filename('default_date_format01.xlsx')

    def test_create_file_user_date_format(self):
        if False:
            i = 10
            return i + 15
        'Test write_datetime with explicit date format.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column(0, 0, 12)
        format1 = workbook.add_format({'num_format': 'yyyy\\-mm\\-dd'})
        date1 = datetime.strptime('2013-07-25', '%Y-%m-%d')
        worksheet.write_datetime(0, 0, date1, format1)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_default_date_format(self):
        if False:
            print('Hello World!')
        'Test write_datetime with default date format.'
        workbook = Workbook(self.got_filename, {'default_date_format': 'yyyy\\-mm\\-dd'})
        worksheet = workbook.add_worksheet()
        worksheet.set_column(0, 0, 12)
        date1 = datetime.strptime('2013-07-25', '%Y-%m-%d')
        worksheet.write_datetime(0, 0, date1)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_default_date_format_write(self):
        if False:
            i = 10
            return i + 15
        'Test write_datetime with default date format.'
        workbook = Workbook(self.got_filename, {'default_date_format': 'yyyy\\-mm\\-dd'})
        worksheet = workbook.add_worksheet()
        worksheet.set_column(0, 0, 12)
        date1 = datetime.strptime('2013-07-25', '%Y-%m-%d')
        worksheet.write('A1', date1)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_default_date_format_write_row(self):
        if False:
            for i in range(10):
                print('nop')
        'Test write_row with default date format.'
        workbook = Workbook(self.got_filename, {'default_date_format': 'yyyy\\-mm\\-dd'})
        worksheet = workbook.add_worksheet()
        worksheet.set_column(0, 0, 12)
        date1 = datetime.strptime('2013-07-25', '%Y-%m-%d')
        worksheet.write_row('A1', [date1])
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_default_date_format_write_column(self):
        if False:
            i = 10
            return i + 15
        'Test write_column with default date format.'
        workbook = Workbook(self.got_filename, {'default_date_format': 'yyyy\\-mm\\-dd'})
        worksheet = workbook.add_worksheet()
        worksheet.set_column(0, 0, 12)
        date1 = datetime.strptime('2013-07-25', '%Y-%m-%d')
        worksheet.write_column(0, 0, [date1])
        workbook.close()
        self.assertExcelEqual()