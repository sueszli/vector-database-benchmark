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
        self.set_filename('escapes01.xlsx')
        self.ignore_files = ['xl/calcChain.xml', '[Content_Types].xml', 'xl/_rels/workbook.xml.rels']

    def test_create_file(self):
        if False:
            return 10
        'Test creation of a file with strings that require XML escaping.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet('5&4')
        worksheet.write_formula(0, 0, '=IF(1>2,0,1)', None, 1)
        worksheet.write_formula(1, 0, '=CONCATENATE("\'","<>&")', None, "'<>&")
        worksheet.write_formula(2, 0, '=1&"b"', None, '1b')
        worksheet.write_formula(3, 0, '="\'"', None, "'")
        worksheet.write_formula(4, 0, '=""""', None, '"')
        worksheet.write_formula(5, 0, '="&" & "&"', None, '&&')
        worksheet.write_string(7, 0, '"&<>')
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_write(self):
        if False:
            return 10
        'Test formulas with write() method.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet('5&4')
        worksheet.write(0, 0, '=IF(1>2,0,1)', None, 1)
        worksheet.write(1, 0, '=CONCATENATE("\'","<>&")', None, "'<>&")
        worksheet.write(2, 0, '=1&"b"', None, '1b')
        worksheet.write(3, 0, '="\'"', None, "'")
        worksheet.write(4, 0, '=""""', None, '"')
        worksheet.write(5, 0, '="&" & "&"', None, '&&')
        worksheet.write_string(7, 0, '"&<>')
        workbook.close()
        self.assertExcelEqual()

    def test_create_file_A1(self):
        if False:
            print('Hello World!')
        'Test formulas with A1 notation.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet('5&4')
        worksheet.write_formula('A1', '=IF(1>2,0,1)', None, 1)
        worksheet.write_formula('A2', '=CONCATENATE("\'","<>&")', None, "'<>&")
        worksheet.write_formula('A3', '=1&"b"', None, '1b')
        worksheet.write_formula('A4', '="\'"', None, "'")
        worksheet.write_formula('A5', '=""""', None, '"')
        worksheet.write_formula('A6', '="&" & "&"', None, '&&')
        worksheet.write_string(7, 0, '"&<>')
        workbook.close()
        self.assertExcelEqual()