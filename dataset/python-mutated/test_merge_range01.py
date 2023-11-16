import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...worksheet import Worksheet
from ...format import Format
from ...sharedstrings import SharedStringTable

class TestAssembleWorksheet(unittest.TestCase):
    """
    Test assembling a complete Worksheet file.

    """

    def test_assemble_xml_file(self):
        if False:
            return 10
        'Test merged cell range'
        self.maxDiff = None
        fh = StringIO()
        worksheet = Worksheet()
        worksheet._set_filehandle(fh)
        worksheet.str_table = SharedStringTable()
        worksheet.select()
        cell_format = Format({'xf_index': 1})
        worksheet.merge_range('B3:C3', 'Foo', cell_format)
        worksheet.select()
        worksheet._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n                  <dimension ref="B3:C3"/>\n                  <sheetViews>\n                    <sheetView tabSelected="1" workbookViewId="0"/>\n                  </sheetViews>\n                  <sheetFormatPr defaultRowHeight="15"/>\n                  <sheetData>\n                    <row r="3" spans="2:3">\n                      <c r="B3" s="1" t="s">\n                        <v>0</v>\n                      </c>\n                      <c r="C3" s="1"/>\n                    </row>\n                  </sheetData>\n                  <mergeCells count="1">\n                    <mergeCell ref="B3:C3"/>\n                  </mergeCells>\n                  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>\n                </worksheet>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)

    def test_assemble_xml_file_write(self):
        if False:
            i = 10
            return i + 15
        'Test writing a worksheet with a blank cell with write() method.'
        self.maxDiff = None
        fh = StringIO()
        worksheet = Worksheet()
        worksheet._set_filehandle(fh)
        cell_format = Format({'xf_index': 1})
        worksheet.write(0, 0, None)
        worksheet.write(1, 2, None, cell_format)
        worksheet.select()
        worksheet._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n                  <dimension ref="C2"/>\n                  <sheetViews>\n                    <sheetView tabSelected="1" workbookViewId="0"/>\n                  </sheetViews>\n                  <sheetFormatPr defaultRowHeight="15"/>\n                  <sheetData>\n                    <row r="2" spans="3:3">\n                      <c r="C2" s="1"/>\n                    </row>\n                  </sheetData>\n                  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>\n                </worksheet>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)

    def test_assemble_xml_file_A1(self):
        if False:
            return 10
        'Test writing a worksheet with a blank cell with A1 notation.'
        self.maxDiff = None
        fh = StringIO()
        worksheet = Worksheet()
        worksheet._set_filehandle(fh)
        cell_format = Format({'xf_index': 1})
        worksheet.write_blank('A1', None)
        worksheet.write_blank('C2', None, cell_format)
        worksheet.select()
        worksheet._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n                  <dimension ref="C2"/>\n                  <sheetViews>\n                    <sheetView tabSelected="1" workbookViewId="0"/>\n                  </sheetViews>\n                  <sheetFormatPr defaultRowHeight="15"/>\n                  <sheetData>\n                    <row r="2" spans="3:3">\n                      <c r="C2" s="1"/>\n                    </row>\n                  </sheetData>\n                  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>\n                </worksheet>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)