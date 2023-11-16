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
        worksheet.add_table('C5:D16', {'banded_rows': 0, 'first_column': 1, 'last_column': 1, 'banded_columns': 1})
        worksheet._prepare_tables(1, {})
        fh = StringIO()
        table = Table()
        table._set_filehandle(fh)
        table._set_properties(worksheet.tables[0])
        table._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" id="1" name="Table1" displayName="Table1" ref="C5:D16" totalsRowShown="0">\n                  <autoFilter ref="C5:D16"/>\n                  <tableColumns count="2">\n                    <tableColumn id="1" name="Column1"/>\n                    <tableColumn id="2" name="Column2"/>\n                  </tableColumns>\n                  <tableStyleInfo name="TableStyleMedium9" showFirstColumn="1" showLastColumn="1" showRowStripes="0" showColumnStripes="1"/>\n                </table>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)