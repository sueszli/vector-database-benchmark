"""Tests for general functionality of the KGML modules."""
import os
import unittest
from Bio.Graphics.ColorSpiral import ColorSpiral
from Bio import MissingExternalDependencyError
try:
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor
except ImportError:
    raise MissingExternalDependencyError('Install reportlab if you want to use Bio.Graphics.') from None
try:
    c = HexColor('#8080F780')
except TypeError:
    raise MissingExternalDependencyError('Install at least reportlab 2.7 for transparency support.') from None
try:
    from PIL import Image
except ImportError:
    raise MissingExternalDependencyError('Install Pillow or its predecessor PIL (Python Imaging Library) if you want to use bitmaps from KGML.') from None
from Bio.KEGG.KGML.KGML_parser import read
from Bio.Graphics.KGML_vis import KGMLCanvas

class PathwayData:
    """Convenience structure for testing pathway data."""

    def __init__(self, name, element_counts, show_pathway_image=False):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.infilename = os.path.join('KEGG', f'ko{name}.xml')
        self.outfilename = os.path.join('KEGG', f'ko{name}.kgml')
        self.element_counts = element_counts
        self.pathway_image = os.path.join('KEGG', f'map{name}.png')
        self.show_pathway_image = show_pathway_image
        self.output_stem = f'Graphics/map{name}'

class KGMLPathwayTest(unittest.TestCase):
    """Import XML file and write KGML.

    Import the ko01100 metabolic map from a local .xml KGML file,
    and write valid KGML output for each.
    """

    def setUp(self):
        if False:
            return 10
        if not os.path.isdir('KEGG'):
            os.mkdir('KEGG')
        self.data = [PathwayData('01100', (3628, 1726, 1746, 149)), PathwayData('03070', (81, 72, 8, 1), True)]
        self.ko_ids = {'ko:K00024', 'ko:K00025', 'ko:K00026', 'ko:K00030', 'ko:K00031', 'ko:K00161', 'ko:K00162', 'ko:K00163', 'ko:K00164', 'ko:K00169', 'ko:K00170', 'ko:K00171', 'ko:K00172', 'ko:K00174', 'ko:K00175', 'ko:K00176', 'ko:K00177', 'ko:K00234', 'ko:K00235', 'ko:K00236', 'ko:K00237', 'ko:K00239', 'ko:K00240', 'ko:K00241', 'ko:K00242', 'ko:K00244', 'ko:K00245', 'ko:K00246', 'ko:K00247', 'ko:K00382', 'ko:K00627', 'ko:K00658', 'ko:K01596', 'ko:K01610', 'ko:K01643', 'ko:K01644', 'ko:K01646', 'ko:K01647', 'ko:K01648', 'ko:K01676', 'ko:K01677', 'ko:K01678', 'ko:K01679', 'ko:K01681', 'ko:K01682', 'ko:K01899', 'ko:K01900', 'ko:K01902', 'ko:K01903', 'ko:K01958', 'ko:K01959', 'ko:K01960'}

    def test_render_KGML_basic(self):
        if False:
            return 10
        'Basic rendering of KGML: write to PDF without modification.'
        for p in self.data:
            with open(p.infilename) as f:
                pathway = read(f)
                pathway.image = p.pathway_image
                kgml_map = KGMLCanvas(pathway)
                kgml_map.import_imagemap = p.show_pathway_image
                kgml_map.draw(p.output_stem + '_original.pdf')

    def test_render_KGML_modify(self):
        if False:
            for i in range(10):
                print('nop')
        'Rendering of KGML to PDF, with modification.'
        p = self.data
        with open(p[0].infilename) as f:
            pathway = read(f)
            mod_rs = [e for e in pathway.orthologs if len(set(e.name.split()).intersection(self.ko_ids))]
            for r in mod_rs:
                for g in r.graphics:
                    g.width = 10
            kgml_map = KGMLCanvas(pathway)
            kgml_map.draw(p[0].output_stem + '_widths.pdf')
        with open(p[1].infilename) as f:
            pathway = read(f)
            orthologs = list(pathway.orthologs)
            cs = ColorSpiral(a=2, b=0.2, v_init=0.85, v_final=0.5, jitter=0.03)
            colors = cs.get_colors(len(orthologs))
            for (o, c) in zip(orthologs, colors):
                for g in o.graphics:
                    g.bgcolor = c
            kgml_map = KGMLCanvas(pathway)
            pathway.image = p[1].pathway_image
            kgml_map.import_imagemap = p[1].show_pathway_image
            kgml_map.draw(p[1].output_stem + '_colors.pdf')

    def test_render_KGML_transparency(self):
        if False:
            for i in range(10):
                print('nop')
        'Rendering of KGML to PDF, with color alpha channel.'
        p = self.data
        with open(p[0].infilename) as f:
            pathway = read(f)
            mod_rs = [e for e in pathway.orthologs if len(set(e.name.split()).intersection(self.ko_ids))]
            for r in mod_rs:
                for g in r.graphics:
                    g.fgcolor = g.fgcolor + '77'
                    g.width = 20
            kgml_map = KGMLCanvas(pathway)
            kgml_map.draw(p[0].output_stem + '_transparency.pdf')
        with open(p[1].infilename) as f:
            pathway = read(f)
            orthologs = list(pathway.orthologs)
            cs = ColorSpiral(a=2, b=0.2, v_init=0.85, v_final=0.5, jitter=0.03)
            colors = cs.get_colors(len(orthologs))
            for (o, c) in zip(orthologs, colors):
                c = c + (0.5,)
                for g in o.graphics:
                    g.bgcolor = c
            kgml_map = KGMLCanvas(pathway)
            pathway.image = p[1].pathway_image
            kgml_map.import_imagemap = p[1].show_pathway_image
            kgml_map.draw(p[1].output_stem + '_transparency.pdf')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)