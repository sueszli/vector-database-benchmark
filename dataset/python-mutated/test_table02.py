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
            return 10
        'Test writing a table'
        self.maxDiff = None
        worksheet = Worksheet()
        worksheet.worksheet_meta = WorksheetMeta()
        worksheet.str_table = SharedStringTable()
        worksheet.add_table('D4:I15', {'style': 'Table Style Light 17'})
        worksheet._prepare_tables(1, {})
        fh = StringIO()
        table = Table()
        table._set_filehandle(fh)
        table._set_properties(worksheet.tables[0])
        table._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" id="1" name="Table1" displayName="Table1" ref="D4:I15" totalsRowShown="0">\n                  <autoFilter ref="D4:I15"/>\n                  <tableColumns count="6">\n                    <tableColumn id="1" name="Column1"/>\n                    <tableColumn id="2" name="Column2"/>\n                    <tableColumn id="3" name="Column3"/>\n                    <tableColumn id="4" name="Column4"/>\n                    <tableColumn id="5" name="Column5"/>\n                    <tableColumn id="6" name="Column6"/>\n                  </tableColumns>\n                  <tableStyleInfo name="TableStyleLight17" showFirstColumn="0" showLastColumn="0" showRowStripes="1" showColumnStripes="0"/>\n                </table>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)