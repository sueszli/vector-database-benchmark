import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...worksheet import Worksheet

class TestAssembleWorksheet(unittest.TestCase):
    """
    Test assembling a complete Worksheet file.

    """

    def test_assemble_xml_file(self):
        if False:
            print('Hello World!')
        'Test writing a worksheet with conditional formatting.'
        self.maxDiff = None
        fh = StringIO()
        worksheet = Worksheet()
        worksheet._set_filehandle(fh)
        worksheet.select()
        worksheet.write('A1', 1)
        worksheet.write('A2', 2)
        worksheet.write('A3', 3)
        worksheet.write('A4', 4)
        worksheet.write('A5', 5)
        worksheet.write('A6', 6)
        worksheet.write('A7', 7)
        worksheet.write('A8', 8)
        worksheet.write('A9', 9)
        worksheet.write('A10', 10)
        worksheet.write('A11', 11)
        worksheet.write('A12', 12)
        worksheet.conditional_format('A1:A12', {'type': 'data_bar', 'min_value': 5, 'mid_value': 52, 'max_value': 90, 'min_length': 5, 'max_length': 95, 'min_type': 'num', 'mid_type': 'percentile', 'max_type': 'percent', 'bar_color': '#8DB4E3'})
        worksheet._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n                  <dimension ref="A1:A12"/>\n                  <sheetViews>\n                    <sheetView tabSelected="1" workbookViewId="0"/>\n                  </sheetViews>\n                  <sheetFormatPr defaultRowHeight="15"/>\n                  <sheetData>\n                    <row r="1" spans="1:1">\n                      <c r="A1">\n                        <v>1</v>\n                      </c>\n                    </row>\n                    <row r="2" spans="1:1">\n                      <c r="A2">\n                        <v>2</v>\n                      </c>\n                    </row>\n                    <row r="3" spans="1:1">\n                      <c r="A3">\n                        <v>3</v>\n                      </c>\n                    </row>\n                    <row r="4" spans="1:1">\n                      <c r="A4">\n                        <v>4</v>\n                      </c>\n                    </row>\n                    <row r="5" spans="1:1">\n                      <c r="A5">\n                        <v>5</v>\n                      </c>\n                    </row>\n                    <row r="6" spans="1:1">\n                      <c r="A6">\n                        <v>6</v>\n                      </c>\n                    </row>\n                    <row r="7" spans="1:1">\n                      <c r="A7">\n                        <v>7</v>\n                      </c>\n                    </row>\n                    <row r="8" spans="1:1">\n                      <c r="A8">\n                        <v>8</v>\n                      </c>\n                    </row>\n                    <row r="9" spans="1:1">\n                      <c r="A9">\n                        <v>9</v>\n                      </c>\n                    </row>\n                    <row r="10" spans="1:1">\n                      <c r="A10">\n                        <v>10</v>\n                      </c>\n                    </row>\n                    <row r="11" spans="1:1">\n                      <c r="A11">\n                        <v>11</v>\n                      </c>\n                    </row>\n                    <row r="12" spans="1:1">\n                      <c r="A12">\n                        <v>12</v>\n                      </c>\n                    </row>\n                  </sheetData>\n                  <conditionalFormatting sqref="A1:A12">\n                    <cfRule type="dataBar" priority="1">\n                      <dataBar minLength="5" maxLength="95">\n                        <cfvo type="num" val="5"/>\n                        <cfvo type="percent" val="90"/>\n                        <color rgb="FF8DB4E3"/>\n                      </dataBar>\n                    </cfRule>\n                  </conditionalFormatting>\n                  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>\n                </worksheet>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)