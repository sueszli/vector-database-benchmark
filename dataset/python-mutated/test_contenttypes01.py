import unittest
from io import StringIO
from ..helperfunctions import _xml_to_list
from ...contenttypes import ContentTypes

class TestAssembleContentTypes(unittest.TestCase):
    """
    Test assembling a complete ContentTypes file.

    """

    def test_assemble_xml_file(self):
        if False:
            return 10
        'Test writing an ContentTypes file.'
        self.maxDiff = None
        fh = StringIO()
        content = ContentTypes()
        content._set_filehandle(fh)
        content._add_worksheet_name('sheet1')
        content._add_default(('jpeg', 'image/jpeg'))
        content._add_shared_strings()
        content._add_calc_chain()
        content._assemble_xml_file()
        exp = _xml_to_list('\n                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n                <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n\n                  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n                  <Default Extension="xml" ContentType="application/xml"/>\n                  <Default Extension="jpeg" ContentType="image/jpeg"/>\n\n                  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>\n                  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>\n                  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>\n                  <Override PartName="/xl/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>\n                  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>\n                  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>\n                  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>\n                  <Override PartName="/xl/calcChain.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.calcChain+xml"/>\n                </Types>\n                ')
        got = _xml_to_list(fh.getvalue())
        self.assertEqual(got, exp)