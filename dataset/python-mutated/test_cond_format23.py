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
            for i in range(10):
                print('nop')
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
        worksheet.conditional_format('A1', {'type': 'icon_set', 'icon_style': '3_arrows_gray'})
        worksheet.conditional_format('A2', {'type': 'icon_set', 'icon_style': '3_traffic_lights'})
        worksheet.conditional_format('A3', {'type': 'icon_set', 'icon_style': '3_signs'})
        worksheet.conditional_format('A4', {'type': 'icon_set', 'icon_style': '3_symbols'})
        worksheet.conditional_format('A5', {'type': 'icon_set', 'icon_style': '4_arrows_gray'})
        worksheet.conditional_format('A6', {'type': 'icon_set', 'icon_style': '4_ratings'})
        worksheet.conditional_format('A7', {'type': 'icon_set', 'icon_style': '5_arrows'})
        worksheet.conditional_format('A8', {'type': 'icon_set', 'icon_style': '5_ratings'})
        worksheet._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n                  <dimension ref="A1:A8"/>\n                  <sheetViews>\n                    <sheetView tabSelected="1" workbookViewId="0"/>\n                  </sheetViews>\n                  <sheetFormatPr defaultRowHeight="15"/>\n                  <sheetData>\n                    <row r="1" spans="1:1">\n                      <c r="A1">\n                        <v>1</v>\n                      </c>\n                    </row>\n                    <row r="2" spans="1:1">\n                      <c r="A2">\n                        <v>2</v>\n                      </c>\n                    </row>\n                    <row r="3" spans="1:1">\n                      <c r="A3">\n                        <v>3</v>\n                      </c>\n                    </row>\n                    <row r="4" spans="1:1">\n                      <c r="A4">\n                        <v>4</v>\n                      </c>\n                    </row>\n                    <row r="5" spans="1:1">\n                      <c r="A5">\n                        <v>5</v>\n                      </c>\n                    </row>\n                    <row r="6" spans="1:1">\n                      <c r="A6">\n                        <v>6</v>\n                      </c>\n                    </row>\n                    <row r="7" spans="1:1">\n                      <c r="A7">\n                        <v>7</v>\n                      </c>\n                    </row>\n                    <row r="8" spans="1:1">\n                      <c r="A8">\n                        <v>8</v>\n                      </c>\n                    </row>\n                  </sheetData>\n                  <conditionalFormatting sqref="A1">\n                    <cfRule type="iconSet" priority="1">\n                      <iconSet iconSet="3ArrowsGray">\n                        <cfvo type="percent" val="0"/>\n                        <cfvo type="percent" val="33"/>\n                        <cfvo type="percent" val="67"/>\n                      </iconSet>\n                    </cfRule>\n                  </conditionalFormatting>\n                  <conditionalFormatting sqref="A2">\n                    <cfRule type="iconSet" priority="2">\n                      <iconSet>\n                        <cfvo type="percent" val="0"/>\n                        <cfvo type="percent" val="33"/>\n                        <cfvo type="percent" val="67"/>\n                      </iconSet>\n                    </cfRule>\n                  </conditionalFormatting>\n                  <conditionalFormatting sqref="A3">\n                    <cfRule type="iconSet" priority="3">\n                      <iconSet iconSet="3Signs">\n                        <cfvo type="percent" val="0"/>\n                        <cfvo type="percent" val="33"/>\n                        <cfvo type="percent" val="67"/>\n                      </iconSet>\n                    </cfRule>\n                  </conditionalFormatting>\n                  <conditionalFormatting sqref="A4">\n                    <cfRule type="iconSet" priority="4">\n                      <iconSet iconSet="3Symbols2">\n                        <cfvo type="percent" val="0"/>\n                        <cfvo type="percent" val="33"/>\n                        <cfvo type="percent" val="67"/>\n                      </iconSet>\n                    </cfRule>\n                  </conditionalFormatting>\n                  <conditionalFormatting sqref="A5">\n                    <cfRule type="iconSet" priority="5">\n                      <iconSet iconSet="4ArrowsGray">\n                        <cfvo type="percent" val="0"/>\n                        <cfvo type="percent" val="25"/>\n                        <cfvo type="percent" val="50"/>\n                        <cfvo type="percent" val="75"/>\n                      </iconSet>\n                    </cfRule>\n                  </conditionalFormatting>\n                  <conditionalFormatting sqref="A6">\n                    <cfRule type="iconSet" priority="6">\n                      <iconSet iconSet="4Rating">\n                        <cfvo type="percent" val="0"/>\n                        <cfvo type="percent" val="25"/>\n                        <cfvo type="percent" val="50"/>\n                        <cfvo type="percent" val="75"/>\n                      </iconSet>\n                    </cfRule>\n                  </conditionalFormatting>\n                  <conditionalFormatting sqref="A7">\n                    <cfRule type="iconSet" priority="7">\n                      <iconSet iconSet="5Arrows">\n                        <cfvo type="percent" val="0"/>\n                        <cfvo type="percent" val="20"/>\n                        <cfvo type="percent" val="40"/>\n                        <cfvo type="percent" val="60"/>\n                        <cfvo type="percent" val="80"/>\n                      </iconSet>\n                    </cfRule>\n                  </conditionalFormatting>\n                  <conditionalFormatting sqref="A8">\n                    <cfRule type="iconSet" priority="8">\n                      <iconSet iconSet="5Rating">\n                        <cfvo type="percent" val="0"/>\n                        <cfvo type="percent" val="20"/>\n                        <cfvo type="percent" val="40"/>\n                        <cfvo type="percent" val="60"/>\n                        <cfvo type="percent" val="80"/>\n                      </iconSet>\n                    </cfRule>\n                  </conditionalFormatting>\n                  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>\n                </worksheet>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)