import unittest
from io import StringIO
from ...vml import Vml

class TestWriteOidmap(unittest.TestCase):
    """
    Test the Vml _write_idmap() method.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.fh = StringIO()
        self.vml = Vml()
        self.vml._set_filehandle(self.fh)

    def test_write_idmap(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_idmap() method'
        self.vml._write_idmap(1)
        exp = '<o:idmap v:ext="edit" data="1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)