import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...table import Table
from ...worksheet import Worksheet
from ...workbook import WorksheetMeta
from ...sharedstrings import SharedStringTable

class TestAssembleTable(unittest.TestCase):
    """
    Test assembling a complete Table file.

    """

    def test_assemble_xml_file(self):
        if False:
            while True:
                i = 10
        'Test writing a table'
        self.maxDiff = None
        worksheet = Worksheet()
        worksheet.worksheet_meta = WorksheetMeta()
        worksheet.str_table = SharedStringTable()
        worksheet.add_table('C3:F13')
        worksheet._prepare_tables(1, {})
        fh = StringIO()
        table = Table()
        table._set_filehandle(fh)
        table._set_properties(worksheet.tables[0])
        table._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" id="1" name="Table1" displayName="Table1" ref="C3:F13" totalsRowShown="0">\n                  <autoFilter ref="C3:F13"/>\n                  <tableColumns count="4">\n                    <tableColumn id="1" name="Column1"/>\n                    <tableColumn id="2" name="Column2"/>\n                    <tableColumn id="3" name="Column3"/>\n                    <tableColumn id="4" name="Column4"/>\n                  </tableColumns>\n                  <tableStyleInfo name="TableStyleMedium9" showFirstColumn="0" showLastColumn="0" showRowStripes="1" showColumnStripes="0"/>\n                </table>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)