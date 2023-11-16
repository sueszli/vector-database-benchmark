import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...comments import Comments

class TestAssembleComments(unittest.TestCase):
    """
    Test assembling a complete Comments file.

    """

    def test_assemble_xml_file(self):
        if False:
            i = 10
            return i + 15
        'Test writing a comments with no cell data.'
        self.maxDiff = None
        fh = StringIO()
        comments = Comments()
        comments._set_filehandle(fh)
        comments._assemble_xml_file([[1, 1, 'Some text', 'John', None, 81, 'Tahoma', 8, 2, [2, 0, 4, 4, 143, 10, 128, 74]]])
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <comments xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n                  <authors>\n                    <author>John</author>\n                  </authors>\n                  <commentList>\n                    <comment ref="B2" authorId="0">\n                      <text>\n                        <r>\n                          <rPr>\n                            <sz val="8"/>\n                            <color indexed="81"/>\n                            <rFont val="Tahoma"/>\n                            <family val="2"/>\n                          </rPr>\n                          <t>Some text</t>\n                        </r>\n                      </text>\n                    </comment>\n                  </commentList>\n                </comments>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)