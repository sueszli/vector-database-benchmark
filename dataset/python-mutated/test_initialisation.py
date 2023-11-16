import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestInitialisation(unittest.TestCase):
    """
    Test initialisation of the Worksheet class and call a method.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_xml_declaration(self):
        if False:
            return 10
        'Test Worksheet xml_declaration()'
        self.worksheet._xml_declaration()
        exp = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)