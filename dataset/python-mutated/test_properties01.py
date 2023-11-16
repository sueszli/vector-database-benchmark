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
        self.set_filename('properties01.xlsx')

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        workbook.set_properties({'title': 'This is an example spreadsheet', 'subject': 'With document properties', 'author': 'Someone', 'manager': 'Dr. Heinz Doofenshmirtz', 'company': 'of Wolves', 'category': 'Example spreadsheets', 'keywords': 'Sample, Example, Properties', 'comments': 'Created with Perl and Excel::Writer::XLSX', 'status': 'Quo'})
        worksheet.set_column('A:A', 70)
        worksheet.write('A1', "Select 'Office Button -> Prepare -> Properties' to see the file properties.")
        workbook.close()
        self.assertExcelEqual()