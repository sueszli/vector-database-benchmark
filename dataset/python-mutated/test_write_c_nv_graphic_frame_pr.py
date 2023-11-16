import unittest
from io import StringIO
from ...drawing import Drawing

class TestWriteXdrcNvGraphicFramePr(unittest.TestCase):
    """
    Test the Drawing _write_c_nv_graphic_frame_pr() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.drawing = Drawing()
        self.drawing._set_filehandle(self.fh)

    def test_write_c_nv_graphic_frame_pr(self):
        if False:
            while True:
                i = 10
        'Test the _write_c_nv_graphic_frame_pr() method'
        self.drawing._write_c_nv_graphic_frame_pr()
        exp = '<xdr:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></xdr:cNvGraphicFramePr>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)