"""Tests for online functionality of the KGML modules."""
import os
import unittest
from Bio import MissingExternalDependencyError
try:
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.pagesizes import A4
except ImportError:
    raise MissingExternalDependencyError('Install reportlab if you want to use Bio.Graphics.') from None
try:
    from PIL import Image
except ImportError:
    raise MissingExternalDependencyError('Install Pillow or its predecessor PIL (Python Imaging Library) if you want to use bitmaps from KGML.') from None
from Bio.KEGG.KGML.KGML_parser import read
from Bio.Graphics.KGML_vis import KGMLCanvas
from test_KGML_graphics import PathwayData
import requires_internet
requires_internet.check()

class KGMLPathwayOnlineTest(unittest.TestCase):
    """Import XML file and write KGML - online tests.

    Import metabolic maps from a local .xml KGML file, and from
    the KEGG site, and write valid KGML output for each
    """

    def setUp(self):
        if False:
            return 10
        if not os.path.isdir('KEGG'):
            os.mkdir('KEGG')
        self.data = [PathwayData('01100', (3628, 1726, 1746, 149)), PathwayData('03070', (81, 72, 8, 1), True)]

    def test_render_KGML_import_map(self):
        if False:
            while True:
                i = 10
        'Basic rendering of KGML: use imported imagemap.\n\n        Uses the URL indicated in the .xml file.\n\n        This test may fail if the imagemap is not available (e.g. if\n        there is not a web connection), and may look odd if the remote\n        imagemap has changed since the local KGML file was downloaded.\n        '
        for p in self.data:
            with open(p.infilename) as f:
                pathway = read(f)
                kgml_map = KGMLCanvas(pathway, import_imagemap=True)
                kgml_map.draw(p.output_stem + '_importmap.pdf')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)