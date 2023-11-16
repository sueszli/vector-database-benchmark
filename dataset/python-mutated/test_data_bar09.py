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
        worksheet.index = 0
        worksheet.conditional_format('A1', {'type': 'data_bar', 'bar_only': True, 'data_bar_2010': True})
        worksheet._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:x14ac="http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac" mc:Ignorable="x14ac">\n                  <dimension ref="A1"/>\n                  <sheetViews>\n                    <sheetView tabSelected="1" workbookViewId="0"/>\n                  </sheetViews>\n                  <sheetFormatPr defaultRowHeight="15" x14ac:dyDescent="0.25"/>\n                  <sheetData/>\n                  <conditionalFormatting sqref="A1">\n                    <cfRule type="dataBar" priority="1">\n                      <dataBar showValue="0">\n                        <cfvo type="min"/>\n                        <cfvo type="max"/>\n                        <color rgb="FF638EC6"/>\n                      </dataBar>\n                      <extLst>\n                        <ext xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main" uri="{B025F937-C7B1-47D3-B67F-A62EFF666E3E}">\n                          <x14:id>{DA7ABA51-AAAA-BBBB-0001-000000000001}</x14:id>\n                        </ext>\n                      </extLst>\n                    </cfRule>\n                  </conditionalFormatting>\n                  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>\n                  <extLst>\n                    <ext xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main" uri="{78C0D931-6437-407d-A8EE-F0AAD7539E65}">\n                      <x14:conditionalFormattings>\n                        <x14:conditionalFormatting xmlns:xm="http://schemas.microsoft.com/office/excel/2006/main">\n                          <x14:cfRule type="dataBar" id="{DA7ABA51-AAAA-BBBB-0001-000000000001}">\n                            <x14:dataBar minLength="0" maxLength="100" border="1" negativeBarBorderColorSameAsPositive="0">\n                              <x14:cfvo type="autoMin"/>\n                              <x14:cfvo type="autoMax"/>\n                              <x14:borderColor rgb="FF638EC6"/>\n                              <x14:negativeFillColor rgb="FFFF0000"/>\n                              <x14:negativeBorderColor rgb="FFFF0000"/>\n                              <x14:axisColor rgb="FF000000"/>\n                            </x14:dataBar>\n                          </x14:cfRule>\n                          <xm:sqref>A1</xm:sqref>\n                        </x14:conditionalFormatting>\n                      </x14:conditionalFormattings>\n                    </ext>\n                  </extLst>\n                </worksheet>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)