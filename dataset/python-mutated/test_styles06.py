import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...styles import Styles
from ...workbook import Workbook

class TestAssembleStyles(unittest.TestCase):
    """
    Test assembling a complete Styles file.

    """

    def test_assemble_xml_file(self):
        if False:
            print('Hello World!')
        'Test for border colour styles.'
        self.maxDiff = None
        fh = StringIO()
        style = Styles()
        style._set_filehandle(fh)
        workbook = Workbook()
        workbook.add_format({'left': 1, 'right': 1, 'top': 1, 'bottom': 1, 'diag_border': 1, 'diag_type': 3, 'left_color': 'red', 'right_color': 'red', 'top_color': 'red', 'bottom_color': 'red', 'diag_color': 'red'})
        workbook._set_default_xf_indices()
        workbook._prepare_format_properties()
        style._set_style_properties([workbook.xf_formats, workbook.palette, workbook.font_count, workbook.num_formats, workbook.border_count, workbook.fill_count, workbook.custom_colors, workbook.dxf_formats, workbook.has_comments])
        style._assemble_xml_file()
        workbook.fileclosed = 1
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n                  <fonts count="1">\n                    <font>\n                      <sz val="11"/>\n                      <color theme="1"/>\n                      <name val="Calibri"/>\n                      <family val="2"/>\n                      <scheme val="minor"/>\n                    </font>\n                  </fonts>\n                  <fills count="2">\n                    <fill>\n                      <patternFill patternType="none"/>\n                    </fill>\n                    <fill>\n                      <patternFill patternType="gray125"/>\n                    </fill>\n                  </fills>\n                  <borders count="2">\n                    <border>\n                      <left/>\n                      <right/>\n                      <top/>\n                      <bottom/>\n                      <diagonal/>\n                    </border>\n                    <border diagonalUp="1" diagonalDown="1">\n                      <left style="thin">\n                        <color rgb="FFFF0000"/>\n                      </left>\n                      <right style="thin">\n                        <color rgb="FFFF0000"/>\n                      </right>\n                      <top style="thin">\n                        <color rgb="FFFF0000"/>\n                      </top>\n                      <bottom style="thin">\n                        <color rgb="FFFF0000"/>\n                      </bottom>\n                      <diagonal style="thin">\n                        <color rgb="FFFF0000"/>\n                      </diagonal>\n                    </border>\n                  </borders>\n                  <cellStyleXfs count="1">\n                    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>\n                  </cellStyleXfs>\n                  <cellXfs count="2">\n                    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>\n                    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1"/>\n                  </cellXfs>\n                  <cellStyles count="1">\n                    <cellStyle name="Normal" xfId="0" builtinId="0"/>\n                  </cellStyles>\n                  <dxfs count="0"/>\n                  <tableStyles count="0" defaultTableStyle="TableStyleMedium9" defaultPivotStyle="PivotStyleLight16"/>\n                </styleSheet>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)