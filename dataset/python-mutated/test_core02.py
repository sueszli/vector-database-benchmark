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
            print('Hello World!')
        'Test writing an Core file.'
        self.maxDiff = None
        fh = StringIO()
        core = Core()
        core._set_filehandle(fh)
        properties = {'title': 'This is an example spreadsheet', 'subject': 'With document properties', 'author': 'John McNamara', 'manager': 'Dr. Heinz Doofenshmirtz', 'company': 'of Wolves', 'category': 'Example spreadsheets', 'keywords': 'Sample, Example, Properties', 'comments': 'Created with Python and XlsxWriter', 'status': 'Quo', 'created': datetime(2011, 4, 6, 19, 45, 15)}
        core._set_properties(properties)
        core._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n                  <dc:title>This is an example spreadsheet</dc:title>\n                  <dc:subject>With document properties</dc:subject>\n                  <dc:creator>John McNamara</dc:creator>\n                  <cp:keywords>Sample, Example, Properties</cp:keywords>\n                  <dc:description>Created with Python and XlsxWriter</dc:description>\n                  <cp:lastModifiedBy>John McNamara</cp:lastModifiedBy>\n                  <dcterms:created xsi:type="dcterms:W3CDTF">2011-04-06T19:45:15Z</dcterms:created>\n                  <dcterms:modified xsi:type="dcterms:W3CDTF">2011-04-06T19:45:15Z</dcterms:modified>\n                  <cp:category>Example spreadsheets</cp:category>\n                  <cp:contentStatus>Quo</cp:contentStatus>\n                </cp:coreProperties>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)