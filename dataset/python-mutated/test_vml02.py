import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list, _vml_to_list
from ...vml import Vml

class TestAssembleVml(unittest.TestCase):
    """
    Test assembling a complete Vml file.

    """

    def test_assemble_xml_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test writing a vml with no cell data.'
        self.maxDiff = None
        fh = StringIO()
        vml = Vml()
        vml._set_filehandle(fh)
        vml._assemble_xml_file(1, 1024, None, [{'row': 1, 'col': 2, 'fillcolor': 'buttonFace [67]', 'vertices': [2, 1, 0, 0, 3, 2, 0, 0, 128, 20, 64, 20], 'font': {'caption': 'Button 1'}, 'macro': '[0]!Button1_Click'}])
        exp = _vml_to_list('\n                <xml xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel">\n                  <o:shapelayout v:ext="edit">\n                    <o:idmap v:ext="edit" data="1"/>\n                  </o:shapelayout>\n                  <v:shapetype id="_x0000_t201" coordsize="21600,21600" o:spt="201" path="m,l,21600r21600,l21600,xe">\n                    <v:stroke joinstyle="miter"/>\n                    <v:path shadowok="f" o:extrusionok="f" strokeok="f" fillok="f" o:connecttype="rect"/>\n                    <o:lock v:ext="edit" shapetype="t"/>\n                  </v:shapetype>\n                  <v:shape id="_x0000_s1025" type="#_x0000_t201" style="position:absolute;margin-left:96pt;margin-top:15pt;width:48pt;height:15pt;z-index:1;mso-wrap-style:tight" o:button="t" fillcolor="buttonFace [67]" strokecolor="windowText [64]" o:insetmode="auto">\n                    <v:fill color2="buttonFace [67]" o:detectmouseclick="t"/>\n                    <o:lock v:ext="edit" rotation="t"/>\n                    <v:textbox style="mso-direction-alt:auto" o:singleclick="f">\n                      <div style="text-align:center">\n                        <font face="Calibri" size="220" color="#000000">Button 1</font>\n                      </div>\n                    </v:textbox>\n                    <x:ClientData ObjectType="Button">\n                      <x:Anchor>2, 0, 1, 0, 3, 0, 2, 0</x:Anchor>\n                      <x:PrintObject>False</x:PrintObject>\n                      <x:AutoFill>False</x:AutoFill>\n                      <x:FmlaMacro>[0]!Button1_Click</x:FmlaMacro>\n                      <x:TextHAlign>Center</x:TextHAlign>\n                      <x:TextVAlign>Center</x:TextVAlign>\n                    </x:ClientData>\n                  </v:shape>\n                </xml>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)