import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...drawing import Drawing

class TestAssembleDrawing(unittest.TestCase):
    """
    Test assembling a complete Drawing file.

    """

    def test_assemble_xml_file(self):
        if False:
            while True:
                i = 10
        'Test writing a drawing with no cell data.'
        self.maxDiff = None
        fh = StringIO()
        drawing = Drawing()
        drawing._set_filehandle(fh)
        dimensions = [4, 8, 457200, 104775, 12, 22, 152400, 180975, 0, 0]
        drawing_object = drawing._add_drawing_object()
        drawing_object['type'] = 1
        drawing_object['dimensions'] = dimensions
        drawing_object['width'] = 0
        drawing_object['height'] = 0
        drawing_object['description'] = None
        drawing_object['shape'] = None
        drawing_object['anchor'] = 1
        drawing_object['rel_index'] = 1
        drawing_object['url_rel_index'] = 0
        drawing_object['tip'] = None
        drawing.embedded = 1
        drawing._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">\n                  <xdr:twoCellAnchor>\n                    <xdr:from>\n                      <xdr:col>4</xdr:col>\n                      <xdr:colOff>457200</xdr:colOff>\n                      <xdr:row>8</xdr:row>\n                      <xdr:rowOff>104775</xdr:rowOff>\n                    </xdr:from>\n                    <xdr:to>\n                      <xdr:col>12</xdr:col>\n                      <xdr:colOff>152400</xdr:colOff>\n                      <xdr:row>22</xdr:row>\n                      <xdr:rowOff>180975</xdr:rowOff>\n                    </xdr:to>\n                    <xdr:graphicFrame macro="">\n                      <xdr:nvGraphicFramePr>\n                        <xdr:cNvPr id="2" name="Chart 1"/>\n                        <xdr:cNvGraphicFramePr/>\n                      </xdr:nvGraphicFramePr>\n                      <xdr:xfrm>\n                        <a:off x="0" y="0"/>\n                        <a:ext cx="0" cy="0"/>\n                      </xdr:xfrm>\n                      <a:graphic>\n                        <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart">\n                          <c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="rId1"/>\n                        </a:graphicData>\n                      </a:graphic>\n                    </xdr:graphicFrame>\n                    <xdr:clientData/>\n                  </xdr:twoCellAnchor>\n                </xdr:wsDr>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)