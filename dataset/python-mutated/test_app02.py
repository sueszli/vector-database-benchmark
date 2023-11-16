import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...app import App

class TestAssembleApp(unittest.TestCase):
    """
    Test assembling a complete App file.

    """

    def test_assemble_xml_file(self):
        if False:
            print('Hello World!')
        'Test writing an App file.'
        self.maxDiff = None
        fh = StringIO()
        app = App()
        app._set_filehandle(fh)
        app._add_part_name('Sheet1')
        app._add_part_name('Sheet2')
        app._add_heading_pair(('Worksheets', 2))
        app._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">\n                  <Application>Microsoft Excel</Application>\n                  <DocSecurity>0</DocSecurity>\n                  <ScaleCrop>false</ScaleCrop>\n                  <HeadingPairs>\n                    <vt:vector size="2" baseType="variant">\n                      <vt:variant>\n                        <vt:lpstr>Worksheets</vt:lpstr>\n                      </vt:variant>\n                      <vt:variant>\n                        <vt:i4>2</vt:i4>\n                      </vt:variant>\n                    </vt:vector>\n                  </HeadingPairs>\n                  <TitlesOfParts>\n                    <vt:vector size="2" baseType="lpstr">\n                      <vt:lpstr>Sheet1</vt:lpstr>\n                      <vt:lpstr>Sheet2</vt:lpstr>\n                    </vt:vector>\n                  </TitlesOfParts>\n                  <Company>\n                  </Company>\n                  <LinksUpToDate>false</LinksUpToDate>\n                  <SharedDoc>false</SharedDoc>\n                  <HyperlinksChanged>false</HyperlinksChanged>\n                  <AppVersion>12.0000</AppVersion>\n                </Properties>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)