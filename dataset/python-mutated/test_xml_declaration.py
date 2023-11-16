import unittest
from io import StringIO
from ...vml import Vml

class TestVmlXmlDeclaration(unittest.TestCase):
    """
    Test initialisation of the Vml class and call a method.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fh = StringIO()
        self.vml = Vml()
        self.vml._set_filehandle(self.fh)

    def test_xml_declaration(self):
        if False:
            print('Hello World!')
        'Test Vml xml_declaration()'
        self.vml._xml_declaration()
        exp = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)