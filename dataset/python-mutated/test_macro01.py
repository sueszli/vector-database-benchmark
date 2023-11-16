from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook
from io import BytesIO

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_filename('macro01.xlsm')

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        workbook.add_vba_project(self.vba_dir + 'vbaProject01.bin')
        worksheet.write('A1', 123)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_in_memory(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename, {'in_memory': True})
        worksheet = workbook.add_worksheet()
        workbook.add_vba_project(self.vba_dir + 'vbaProject01.bin')
        worksheet.write('A1', 123)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_bytes_io(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        vba_file = open(self.vba_dir + 'vbaProject01.bin', 'rb')
        vba_data = BytesIO(vba_file.read())
        vba_file.close()
        workbook.add_vba_project(vba_data, True)
        worksheet.write('A1', 123)
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_bytes_io_in_memory(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename, {'in_memory': True})
        worksheet = workbook.add_worksheet()
        vba_file = open(self.vba_dir + 'vbaProject01.bin', 'rb')
        vba_data = BytesIO(vba_file.read())
        vba_file.close()
        workbook.add_vba_project(vba_data, True)
        worksheet.write('A1', 123)
        workbook.close()
        self.assertExcelEqual()