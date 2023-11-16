import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...relationships import Relationships

class TestAssembleRelationships(unittest.TestCase):
    """
    Test assembling a complete Relationships file.

    """

    def test_assemble_xml_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test writing an Relationships file.'
        self.maxDiff = None
        fh = StringIO()
        rels = Relationships()
        rels._set_filehandle(fh)
        rels._add_document_relationship('/worksheet', 'worksheets/sheet1.xml')
        rels._add_document_relationship('/theme', 'theme/theme1.xml')
        rels._add_document_relationship('/styles', 'styles.xml')
        rels._add_document_relationship('/sharedStrings', 'sharedStrings.xml')
        rels._add_document_relationship('/calcChain', 'calcChain.xml')
        rels._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n                  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>\n                  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>\n                  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>\n                  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>\n                  <Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/calcChain" Target="calcChain.xml"/>\n                </Relationships>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)