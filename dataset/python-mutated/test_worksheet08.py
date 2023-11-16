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
            print('Hello World!')
        'Test writing a worksheet with an array formulas in cells.'
        self.maxDiff = None
        fh = StringIO()
        worksheet = Worksheet()
        worksheet._set_filehandle(fh)
        worksheet.str_table = SharedStringTable()
        worksheet.select()
        worksheet.write_array_formula(0, 0, 2, 0, '{=SUM(B1:C1*B2:C2)}')
        worksheet.write_number(0, 1, 0)
        worksheet.write_number(1, 1, 0)
        worksheet.write_number(2, 1, 0)
        worksheet.write_number(0, 2, 0)
        worksheet.write_number(1, 2, 0)
        worksheet.write_number(2, 2, 0)
        worksheet._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n                  <dimension ref="A1:C3"/>\n                  <sheetViews>\n                    <sheetView tabSelected="1" workbookViewId="0"/>\n                  </sheetViews>\n                  <sheetFormatPr defaultRowHeight="15"/>\n                  <sheetData>\n                    <row r="1" spans="1:3">\n                      <c r="A1">\n                        <f t="array" ref="A1:A3">SUM(B1:C1*B2:C2)</f>\n                        <v>0</v>\n                      </c>\n                      <c r="B1">\n                        <v>0</v>\n                      </c>\n                      <c r="C1">\n                        <v>0</v>\n                      </c>\n                    </row>\n                    <row r="2" spans="1:3">\n                      <c r="A2">\n                        <v>0</v>\n                      </c>\n                      <c r="B2">\n                        <v>0</v>\n                      </c>\n                      <c r="C2">\n                        <v>0</v>\n                      </c>\n                    </row>\n                    <row r="3" spans="1:3">\n                      <c r="A3">\n                        <v>0</v>\n                      </c>\n                      <c r="B3">\n                        <v>0</v>\n                      </c>\n                      <c r="C3">\n                        <v>0</v>\n                      </c>\n                    </row>\n                  </sheetData>\n                  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>\n                </worksheet>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)