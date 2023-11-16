import unittest
from io import StringIO
from ...drawing import Drawing

class TestWriteCchart(unittest.TestCase):
    """
    Test the Drawing _write_c_chart() method.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.fh = StringIO()
        self.drawing = Drawing()
        self.drawing._set_filehandle(self.fh)

    def test_write_c_chart(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_c_chart() method'
        self.drawing._write_c_chart('rId1')
        exp = '<c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="rId1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)