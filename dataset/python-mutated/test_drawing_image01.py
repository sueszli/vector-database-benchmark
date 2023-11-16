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
            for i in range(10):
                print('nop')
        'Test writing a drawing with no cell data.'
        self.maxDiff = None
        fh = StringIO()
        drawing = Drawing()
        drawing._set_filehandle(fh)
        dimensions = [2, 1, 0, 0, 3, 6, 533257, 190357, 1219200, 190500, 0, 0]
        drawing_object = drawing._add_drawing_object()
        drawing_object['type'] = 2
        drawing_object['dimensions'] = dimensions
        drawing_object['width'] = 1142857
        drawing_object['height'] = 1142857
        drawing_object['description'] = 'republic.png'
        drawing_object['shape'] = None
        drawing_object['anchor'] = 2
        drawing_object['rel_index'] = 1
        drawing_object['url_rel_index'] = 0
        drawing_object['tip'] = None
        drawing.embedded = 1
        drawing._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">\n                  <xdr:twoCellAnchor editAs="oneCell">\n                    <xdr:from>\n                      <xdr:col>2</xdr:col>\n                      <xdr:colOff>0</xdr:colOff>\n                      <xdr:row>1</xdr:row>\n                      <xdr:rowOff>0</xdr:rowOff>\n                    </xdr:from>\n                    <xdr:to>\n                      <xdr:col>3</xdr:col>\n                      <xdr:colOff>533257</xdr:colOff>\n                      <xdr:row>6</xdr:row>\n                      <xdr:rowOff>190357</xdr:rowOff>\n                    </xdr:to>\n                    <xdr:pic>\n                      <xdr:nvPicPr>\n                        <xdr:cNvPr id="2" name="Picture 1" descr="republic.png"/>\n                        <xdr:cNvPicPr>\n                          <a:picLocks noChangeAspect="1"/>\n                        </xdr:cNvPicPr>\n                      </xdr:nvPicPr>\n                      <xdr:blipFill>\n                        <a:blip xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:embed="rId1"/>\n                        <a:stretch>\n                          <a:fillRect/>\n                        </a:stretch>\n                      </xdr:blipFill>\n                      <xdr:spPr>\n                        <a:xfrm>\n                          <a:off x="1219200" y="190500"/>\n                          <a:ext cx="1142857" cy="1142857"/>\n                        </a:xfrm>\n                        <a:prstGeom prst="rect">\n                          <a:avLst/>\n                        </a:prstGeom>\n                      </xdr:spPr>\n                    </xdr:pic>\n                    <xdr:clientData/>\n                  </xdr:twoCellAnchor>\n                </xdr:wsDr>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)

    def test_assemble_xml_file_with_url(self):
        if False:
            i = 10
            return i + 15
        'Test writing a drawing with no cell data.'
        self.maxDiff = None
        fh = StringIO()
        drawing = Drawing()
        drawing._set_filehandle(fh)
        drawing = Drawing()
        drawing._set_filehandle(fh)
        dimensions = [2, 1, 0, 0, 3, 6, 533257, 190357, 1219200, 190500, 0, 0]
        drawing_object = drawing._add_drawing_object()
        drawing_object['type'] = 2
        drawing_object['dimensions'] = dimensions
        drawing_object['width'] = 1142857
        drawing_object['height'] = 1142857
        drawing_object['description'] = 'republic.png'
        drawing_object['shape'] = None
        drawing_object['anchor'] = 2
        drawing_object['rel_index'] = 2
        drawing_object['url_rel_index'] = 1
        drawing_object['tip'] = 'this is a tooltip'
        drawing.embedded = 1
        drawing._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">\n                  <xdr:twoCellAnchor editAs="oneCell">\n                    <xdr:from>\n                      <xdr:col>2</xdr:col>\n                      <xdr:colOff>0</xdr:colOff>\n                      <xdr:row>1</xdr:row>\n                      <xdr:rowOff>0</xdr:rowOff>\n                    </xdr:from>\n                    <xdr:to>\n                      <xdr:col>3</xdr:col>\n                      <xdr:colOff>533257</xdr:colOff>\n                      <xdr:row>6</xdr:row>\n                      <xdr:rowOff>190357</xdr:rowOff>\n                    </xdr:to>\n                    <xdr:pic>\n                    <xdr:nvPicPr>\n                        <xdr:cNvPr id="2" name="Picture 1" descr="republic.png">\n                          <a:hlinkClick xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="rId1" tooltip="this is a tooltip"/>\n                        </xdr:cNvPr>\n                        <xdr:cNvPicPr>\n                            <a:picLocks noChangeAspect="1"/>\n                        </xdr:cNvPicPr>\n                    </xdr:nvPicPr>\n                    <xdr:blipFill>\n                        <a:blip xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:embed="rId2"/>\n                        <a:stretch>\n                            <a:fillRect/>\n                        </a:stretch>\n                    </xdr:blipFill>\n                      <xdr:spPr>\n                        <a:xfrm>\n                          <a:off x="1219200" y="190500"/>\n                          <a:ext cx="1142857" cy="1142857"/>\n                        </a:xfrm>\n                        <a:prstGeom prst="rect">\n                          <a:avLst/>\n                        </a:prstGeom>\n                      </xdr:spPr>\n                    </xdr:pic>\n                    <xdr:clientData/>\n                  </xdr:twoCellAnchor>\n                </xdr:wsDr>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)