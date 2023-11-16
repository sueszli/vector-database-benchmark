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
            return 10
        'Test writing a vml with no cell data.'
        self.maxDiff = None
        fh = StringIO()
        vml = Vml()
        vml._set_filehandle(fh)
        vml._assemble_xml_file(1, 1024, [[1, 1, 'Some text', '', None, '#ffffe1', 'Tahoma', 8, 2, [2, 0, 15, 10, 4, 4, 15, 4, 143, 10, 128, 74]]], [])
        exp = _vml_to_list('\n                <xml xmlns:v="urn:schemas-microsoft-com:vml"\n                 xmlns:o="urn:schemas-microsoft-com:office:office"\n                 xmlns:x="urn:schemas-microsoft-com:office:excel">\n                 <o:shapelayout v:ext="edit">\n                  <o:idmap v:ext="edit" data="1"/>\n                 </o:shapelayout><v:shapetype id="_x0000_t202" coordsize="21600,21600" o:spt="202"\n                  path="m,l,21600r21600,l21600,xe">\n                  <v:stroke joinstyle="miter"/>\n                  <v:path gradientshapeok="t" o:connecttype="rect"/>\n                 </v:shapetype><v:shape id="_x0000_s1025" type="#_x0000_t202" style=\'position:absolute;\n                  margin-left:107.25pt;margin-top:7.5pt;width:96pt;height:55.5pt;z-index:1;\n                  visibility:hidden\' fillcolor="#ffffe1" o:insetmode="auto">\n                  <v:fill color2="#ffffe1"/>\n                  <v:shadow on="t" color="black" obscured="t"/>\n                  <v:path o:connecttype="none"/>\n                  <v:textbox style=\'mso-direction-alt:auto\'>\n                   <div style=\'text-align:left\'></div>\n                  </v:textbox>\n                  <x:ClientData ObjectType="Note">\n                   <x:MoveWithCells/>\n                   <x:SizeWithCells/>\n                   <x:Anchor>\n                    2, 15, 0, 10, 4, 15, 4, 4</x:Anchor>\n                   <x:AutoFill>False</x:AutoFill>\n                   <x:Row>1</x:Row>\n                   <x:Column>1</x:Column>\n                  </x:ClientData>\n                 </v:shape></xml>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)