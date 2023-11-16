import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...worksheet import Worksheet
from ...format import Format

class TestAssembleWorksheet(unittest.TestCase):
    """
    Test assembling a complete Worksheet file.

    """

    def test_assemble_xml_file(self):
        if False:
            print('Hello World!')
        'Test writing a worksheet with row formatting set.'
        self.maxDiff = None
        fh = StringIO()
        worksheet = Worksheet()
        worksheet._set_filehandle(fh)
        cell_format = Format({'xf_index': 1})
        worksheet.set_row(1, 30)
        worksheet.set_row(3, None, None, {'hidden': 1})
        worksheet.set_row(6, None, cell_format)
        worksheet.set_row(9, 3)
        worksheet.set_row(12, 24, None, {'hidden': 1})
        worksheet.set_row(14, 0)
        worksheet.select()
        worksheet._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n                  <dimension ref="A2:A15"/>\n                  <sheetViews>\n                    <sheetView tabSelected="1" workbookViewId="0"/>\n                  </sheetViews>\n                  <sheetFormatPr defaultRowHeight="15"/>\n                  <sheetData>\n                    <row r="2" ht="30" customHeight="1"/>\n                    <row r="4" hidden="1"/>\n                    <row r="7" s="1" customFormat="1"/>\n                    <row r="10" ht="3" customHeight="1"/>\n                    <row r="13" ht="24" hidden="1" customHeight="1"/>\n                    <row r="15" hidden="1"/>\n                  </sheetData>\n                  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>\n                </worksheet>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)