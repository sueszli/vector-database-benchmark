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
        vml._assemble_xml_file(1, 1024, None, None, [[32, 32, 'red', 'CH', 96, 96, 1]])
        exp = _vml_to_list('\n                <xml xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel">\n                  <o:shapelayout v:ext="edit">\n                    <o:idmap v:ext="edit" data="1"/>\n                  </o:shapelayout>\n                  <v:shapetype id="_x0000_t75" coordsize="21600,21600" o:spt="75" o:preferrelative="t" path="m@4@5l@4@11@9@11@9@5xe" filled="f" stroked="f">\n                    <v:stroke joinstyle="miter"/>\n                    <v:formulas>\n                      <v:f eqn="if lineDrawn pixelLineWidth 0"/>\n                      <v:f eqn="sum @0 1 0"/>\n                      <v:f eqn="sum 0 0 @1"/>\n                      <v:f eqn="prod @2 1 2"/>\n                      <v:f eqn="prod @3 21600 pixelWidth"/>\n                      <v:f eqn="prod @3 21600 pixelHeight"/>\n                      <v:f eqn="sum @0 0 1"/>\n                      <v:f eqn="prod @6 1 2"/>\n                      <v:f eqn="prod @7 21600 pixelWidth"/>\n                      <v:f eqn="sum @8 21600 0"/>\n                      <v:f eqn="prod @7 21600 pixelHeight"/>\n                      <v:f eqn="sum @10 21600 0"/>\n                    </v:formulas>\n                    <v:path o:extrusionok="f" gradientshapeok="t" o:connecttype="rect"/>\n                    <o:lock v:ext="edit" aspectratio="t"/>\n                  </v:shapetype>\n                  <v:shape id="CH" o:spid="_x0000_s1025" type="#_x0000_t75" style="position:absolute;margin-left:0;margin-top:0;width:24pt;height:24pt;z-index:1">\n                    <v:imagedata o:relid="rId1" o:title="red"/>\n                    <o:lock v:ext="edit" rotation="t"/>\n                  </v:shape>\n                </xml>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)