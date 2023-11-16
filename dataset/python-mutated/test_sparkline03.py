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
            return 10
        'Test writing a worksheet with no cell data.'
        self.maxDiff = None
        fh = StringIO()
        worksheet = Worksheet()
        worksheet._set_filehandle(fh)
        worksheet.select()
        worksheet.name = 'Sheet1'
        worksheet.excel_version = 2010
        data = [-2, 2, 3, -1, 0]
        worksheet.write_row('A1', data)
        worksheet.write_row('A2', data)
        worksheet.add_sparkline('F1', {'range': 'Sheet1!A1:E1'})
        worksheet.add_sparkline('F2', {'range': 'Sheet1!A2:E2'})
        worksheet._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:x14ac="http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac" mc:Ignorable="x14ac">\n                  <dimension ref="A1:E2"/>\n                  <sheetViews>\n                    <sheetView tabSelected="1" workbookViewId="0"/>\n                  </sheetViews>\n                  <sheetFormatPr defaultRowHeight="15" x14ac:dyDescent="0.25"/>\n                  <sheetData>\n                    <row r="1" spans="1:5" x14ac:dyDescent="0.25">\n                      <c r="A1">\n                        <v>-2</v>\n                      </c>\n                      <c r="B1">\n                        <v>2</v>\n                      </c>\n                      <c r="C1">\n                        <v>3</v>\n                      </c>\n                      <c r="D1">\n                        <v>-1</v>\n                      </c>\n                      <c r="E1">\n                        <v>0</v>\n                      </c>\n                    </row>\n                    <row r="2" spans="1:5" x14ac:dyDescent="0.25">\n                      <c r="A2">\n                        <v>-2</v>\n                      </c>\n                      <c r="B2">\n                        <v>2</v>\n                      </c>\n                      <c r="C2">\n                        <v>3</v>\n                      </c>\n                      <c r="D2">\n                        <v>-1</v>\n                      </c>\n                      <c r="E2">\n                        <v>0</v>\n                      </c>\n                    </row>\n                  </sheetData>\n                  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>\n                  <extLst>\n                    <ext xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main" uri="{05C60535-1F16-4fd2-B633-F4F36F0B64E0}">\n                      <x14:sparklineGroups xmlns:xm="http://schemas.microsoft.com/office/excel/2006/main">\n                        <x14:sparklineGroup displayEmptyCellsAs="gap">\n                          <x14:colorSeries theme="4" tint="-0.499984740745262"/>\n                          <x14:colorNegative theme="5"/>\n                          <x14:colorAxis rgb="FF000000"/>\n                          <x14:colorMarkers theme="4" tint="-0.499984740745262"/>\n                          <x14:colorFirst theme="4" tint="0.39997558519241921"/>\n                          <x14:colorLast theme="4" tint="0.39997558519241921"/>\n                          <x14:colorHigh theme="4"/>\n                          <x14:colorLow theme="4"/>\n                          <x14:sparklines>\n                            <x14:sparkline>\n                              <xm:f>Sheet1!A2:E2</xm:f>\n                              <xm:sqref>F2</xm:sqref>\n                            </x14:sparkline>\n                          </x14:sparklines>\n                        </x14:sparklineGroup>\n                        <x14:sparklineGroup displayEmptyCellsAs="gap">\n                          <x14:colorSeries theme="4" tint="-0.499984740745262"/>\n                          <x14:colorNegative theme="5"/>\n                          <x14:colorAxis rgb="FF000000"/>\n                          <x14:colorMarkers theme="4" tint="-0.499984740745262"/>\n                          <x14:colorFirst theme="4" tint="0.39997558519241921"/>\n                          <x14:colorLast theme="4" tint="0.39997558519241921"/>\n                          <x14:colorHigh theme="4"/>\n                          <x14:colorLow theme="4"/>\n                          <x14:sparklines>\n                            <x14:sparkline>\n                              <xm:f>Sheet1!A1:E1</xm:f>\n                              <xm:sqref>F1</xm:sqref>\n                            </x14:sparkline>\n                          </x14:sparklines>\n                        </x14:sparklineGroup>\n                      </x14:sparklineGroups>\n                    </ext>\n                  </extLst>\n                </worksheet>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)