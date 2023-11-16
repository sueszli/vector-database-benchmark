import unittest
from io import StringIO
from datetime import datetime
from ..helperfunctions import _xml_to_list
from ...core import Core

class TestAssembleCore(unittest.TestCase):
    """
    Test assembling a complete Core file.

    """

    def test_assemble_xml_file(self):
        if False:
            while True:
                i = 10
        'Test writing an Core file.'
        self.maxDiff = None
        fh = StringIO()
        core = Core()
        core._set_filehandle(fh)
        properties = {'author': 'A User', 'created': datetime(2010, 1, 1, 0, 0, 0)}
        core._set_properties(properties)
        core._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n                  <dc:creator>A User</dc:creator>\n                  <cp:lastModifiedBy>A User</cp:lastModifiedBy>\n                  <dcterms:created xsi:type="dcterms:W3CDTF">2010-01-01T00:00:00Z</dcterms:created>\n                  <dcterms:modified xsi:type="dcterms:W3CDTF">2010-01-01T00:00:00Z</dcterms:modified>\n                </cp:coreProperties>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)