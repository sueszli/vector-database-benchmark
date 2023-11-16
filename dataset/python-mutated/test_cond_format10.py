import unittest
from datetime import datetime
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...worksheet import Worksheet

class TestAssembleWorksheet(unittest.TestCase):
    """
    Test assembling a complete Worksheet file.

    """

    def test_assemble_xml_file(self):
        if False:
            i = 10
            return i + 15
        'Test writing a worksheet with conditional formatting.'
        self.maxDiff = None
        fh = StringIO()
        worksheet = Worksheet()
        worksheet._set_filehandle(fh)
        worksheet.select()
        worksheet.write('A1', 10)
        worksheet.write('A2', 20)
        worksheet.write('A3', 30)
        worksheet.write('A4', 40)
        date = datetime.strptime('2011-01-01', '%Y-%m-%d')
        worksheet.conditional_format('A1:A4', {'type': 'date', 'criteria': 'greater than', 'value': date, 'format': None})
        worksheet._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n                  <dimension ref="A1:A4"/>\n                  <sheetViews>\n                    <sheetView tabSelected="1" workbookViewId="0"/>\n                  </sheetViews>\n                  <sheetFormatPr defaultRowHeight="15"/>\n                  <sheetData>\n                    <row r="1" spans="1:1">\n                      <c r="A1">\n                        <v>10</v>\n                      </c>\n                    </row>\n                    <row r="2" spans="1:1">\n                      <c r="A2">\n                        <v>20</v>\n                      </c>\n                    </row>\n                    <row r="3" spans="1:1">\n                      <c r="A3">\n                        <v>30</v>\n                      </c>\n                    </row>\n                    <row r="4" spans="1:1">\n                      <c r="A4">\n                        <v>40</v>\n                      </c>\n                    </row>\n                  </sheetData>\n                  <conditionalFormatting sqref="A1:A4">\n                    <cfRule type="cellIs" priority="1" operator="greaterThan">\n                      <formula>40544</formula>\n                    </cfRule>\n                  </conditionalFormatting>\n                  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>\n                </worksheet>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)