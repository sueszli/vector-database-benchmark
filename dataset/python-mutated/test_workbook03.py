import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...workbook import Workbook

class TestAssembleWorkbook(unittest.TestCase):
    """
    Test assembling a complete Workbook file.

    """

    def test_assemble_xml_file(self):
        if False:
            i = 10
            return i + 15
        'Test writing a workbook with user specified names.'
        self.maxDiff = None
        fh = StringIO()
        workbook = Workbook()
        workbook._set_filehandle(fh)
        workbook.add_worksheet('Non Default Name')
        workbook.add_worksheet('Another Name')
        workbook._assemble_xml_file()
        workbook.fileclosed = 1
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n                  <fileVersion appName="xl" lastEdited="4" lowestEdited="4" rupBuild="4505"/>\n                  <workbookPr defaultThemeVersion="124226"/>\n                  <bookViews>\n                    <workbookView xWindow="240" yWindow="15" windowWidth="16095" windowHeight="9660"/>\n                  </bookViews>\n                  <sheets>\n                    <sheet name="Non Default Name" sheetId="1" r:id="rId1"/>\n                    <sheet name="Another Name" sheetId="2" r:id="rId2"/>\n                  </sheets>\n                  <calcPr calcId="124519" fullCalcOnLoad="1"/>\n                </workbook>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)