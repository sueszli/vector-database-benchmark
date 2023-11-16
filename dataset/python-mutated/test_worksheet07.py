import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...worksheet import Worksheet
from ...sharedstrings import SharedStringTable

class TestAssembleWorksheet(unittest.TestCase):
    """
    Test assembling a complete Worksheet file.

    """

    def test_assemble_xml_file(self):
        if False:
            while True:
                i = 10
        'Test writing a worksheet with formulas in cells.'
        self.maxDiff = None
        fh = StringIO()
        worksheet = Worksheet()
        worksheet._set_filehandle(fh)
        worksheet.str_table = SharedStringTable()
        worksheet.select()
        worksheet.write_number(0, 0, 1)
        worksheet.write_number(1, 0, 2)
        worksheet.write_formula(2, 2, '=A1+A2', None, 3)
        worksheet.write_formula(4, 1, '="<&>" & ";"" \'"', None, '<&>;" \'')
        worksheet._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n                  <dimension ref="A1:C5"/>\n                  <sheetViews>\n                    <sheetView tabSelected="1" workbookViewId="0"/>\n                  </sheetViews>\n                  <sheetFormatPr defaultRowHeight="15"/>\n                  <sheetData>\n                    <row r="1" spans="1:3">\n                      <c r="A1">\n                        <v>1</v>\n                      </c>\n                    </row>\n                    <row r="2" spans="1:3">\n                      <c r="A2">\n                        <v>2</v>\n                      </c>\n                    </row>\n                    <row r="3" spans="1:3">\n                      <c r="C3">\n                        <f>A1+A2</f>\n                        <v>3</v>\n                      </c>\n                    </row>\n                    <row r="5" spans="1:3">\n                      <c r="B5" t="str">\n                        <f>"&lt;&amp;&gt;" &amp; ";"" \'"</f>\n                        <v>&lt;&amp;&gt;;" \'</v>\n                      </c>\n                    </row>\n                  </sheetData>\n                  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>\n                </worksheet>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)