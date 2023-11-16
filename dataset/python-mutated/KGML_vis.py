"""Classes and functions to visualise a KGML Pathway Map.

The KGML definition is as of release KGML v0.7.1
(http://www.kegg.jp/kegg/xml/docs/)

Classes:
"""
import os
import tempfile
from io import BytesIO
try:
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install reportlab if you want to use KGML_vis.') from None
try:
    from PIL import Image
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install pillow if you want to use KGML_vis.') from None
from urllib.request import urlopen
from Bio.KEGG.KGML.KGML_pathway import Pathway

def darken(color, factor=0.7):
    if False:
        for i in range(10):
            print('nop')
    'Return darkened color as a ReportLab RGB color.\n\n    Take a passed color and returns a Reportlab color that is darker by the\n    factor indicated in the parameter.\n    '
    newcol = color_to_reportlab(color)
    for a in ['red', 'green', 'blue']:
        setattr(newcol, a, factor * getattr(newcol, a))
    return newcol

def color_to_reportlab(color):
    if False:
        print('Hello World!')
    'Return the passed color in Reportlab Color format.\n\n    We allow colors to be specified as hex values, tuples, or Reportlab Color\n    objects, and with or without an alpha channel. This function acts as a\n    Rosetta stone for conversion of those formats to a Reportlab Color\n    object, with alpha value.\n\n    Any other color specification is returned directly\n    '
    if isinstance(color, colors.Color):
        return color
    elif isinstance(color, str):
        if color.startswith('0x'):
            color.replace('0x', '#')
        if len(color) == 7:
            return colors.HexColor(color)
        else:
            try:
                return colors.HexColor(color, hasAlpha=True)
            except TypeError:
                raise RuntimeError('Your reportlab seems to be too old, try 2.7 onwards') from None
    elif isinstance(color, tuple):
        return colors.Color(*color)
    return color

def get_temp_imagefilename(url):
    if False:
        return 10
    'Return filename of temporary file containing downloaded image.\n\n    Create a new temporary file to hold the image file at the passed URL\n    and return the filename.\n    '
    img = urlopen(url).read()
    im = Image.open(BytesIO(img))
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fname = f.name
    f.close()
    im.save(fname, 'PNG')
    return fname

class KGMLCanvas:
    """Reportlab Canvas-based representation of a KGML pathway map."""

    def __init__(self, pathway, import_imagemap=False, label_compounds=True, label_orthologs=True, label_reaction_entries=True, label_maps=True, show_maps=False, fontname='Helvetica', fontsize=6, draw_relations=True, show_orthologs=True, show_compounds=True, show_genes=True, show_reaction_entries=True, margins=(0.02, 0.02)):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.pathway = pathway
        self.show_maps = show_maps
        self.show_orthologs = show_orthologs
        self.show_compounds = show_compounds
        self.show_genes = show_genes
        self.show_reaction_entries = show_reaction_entries
        self.label_compounds = label_compounds
        self.label_orthologs = label_orthologs
        self.label_reaction_entries = label_reaction_entries
        self.label_maps = label_maps
        self.fontname = fontname
        self.fontsize = fontsize
        self.draw_relations = draw_relations
        self.non_reactant_transparency = 0.3
        self.import_imagemap = import_imagemap
        self.margins = margins

    def draw(self, filename):
        if False:
            while True:
                i = 10
        'Add the map elements to the drawing.'
        if self.import_imagemap:
            if os.path.isfile(self.pathway.image):
                imfilename = self.pathway.image
            else:
                imfilename = get_temp_imagefilename(self.pathway.image)
            im = Image.open(imfilename)
            (cwidth, cheight) = im.size
        else:
            (cwidth, cheight) = (self.pathway.bounds[1][0], self.pathway.bounds[1][1])
        self.drawing = canvas.Canvas(filename, bottomup=0, pagesize=(cwidth * (1 + 2 * self.margins[0]), cheight * (1 + 2 * self.margins[1])))
        self.drawing.setFont(self.fontname, self.fontsize)
        self.drawing.translate(self.margins[0] * self.pathway.bounds[1][0], self.margins[1] * self.pathway.bounds[1][1])
        if self.import_imagemap:
            self.drawing.saveState()
            self.drawing.scale(1, -1)
            self.drawing.translate(0, -cheight)
            self.drawing.drawImage(imfilename, 0, 0)
            self.drawing.restoreState()
        if self.show_maps:
            self.__add_maps()
        if self.show_reaction_entries:
            self.__add_reaction_entries()
        if self.show_orthologs:
            self.__add_orthologs()
        if self.show_compounds:
            self.__add_compounds()
        if self.show_genes:
            self.__add_genes()
        self.drawing.save()

    def __add_maps(self):
        if False:
            while True:
                i = 10
        "Add maps to the drawing of the map (PRIVATE).\n\n        We do this first, as they're regional labels to be overlaid by\n        information.  Also, we want to set the color to something subtle.\n\n        We're using Hex colors because that's what KGML uses, and\n        Reportlab doesn't mind.\n        "
        for m in self.pathway.maps:
            for g in m.graphics:
                self.drawing.setStrokeColor('#888888')
                self.drawing.setFillColor('#DDDDDD')
                self.__add_graphics(g)
                if self.label_maps:
                    self.drawing.setFillColor('#888888')
                    self.__add_labels(g)

    def __add_graphics(self, graphics):
        if False:
            return 10
        'Add the passed graphics object to the map (PRIVATE).\n\n        Add text, add after the graphics object, for sane Z-ordering.\n        '
        if graphics.type == 'line':
            p = self.drawing.beginPath()
            (x, y) = graphics.coords[0]
            if graphics.width is not None:
                self.drawing.setLineWidth(graphics.width)
            else:
                self.drawing.setLineWidth(1)
            p.moveTo(x, y)
            for (x, y) in graphics.coords:
                p.lineTo(x, y)
            self.drawing.drawPath(p)
            self.drawing.setLineWidth(1)
        if graphics.type == 'circle':
            self.drawing.circle(graphics.x, graphics.y, graphics.width * 0.5, stroke=1, fill=1)
        elif graphics.type == 'roundrectangle':
            self.drawing.roundRect(graphics.x - graphics.width * 0.5, graphics.y - graphics.height * 0.5, graphics.width, graphics.height, min(graphics.width, graphics.height) * 0.1, stroke=1, fill=1)
        elif graphics.type == 'rectangle':
            self.drawing.rect(graphics.x - graphics.width * 0.5, graphics.y - graphics.height * 0.5, graphics.width, graphics.height, stroke=1, fill=1)

    def __add_labels(self, graphics):
        if False:
            while True:
                i = 10
        "Add labels for the passed graphics objects to the map (PRIVATE).\n\n        We don't check that the labels fit inside objects such as circles/\n        rectangles/roundrectangles.\n        "
        if graphics.type == 'line':
            mid_idx = len(graphics.coords) * 0.5
            if int(mid_idx) != mid_idx:
                (idx1, idx2) = (int(mid_idx - 0.5), int(mid_idx + 0.5))
            else:
                (idx1, idx2) = (int(mid_idx - 1), int(mid_idx))
            (x1, y1) = graphics.coords[idx1]
            (x2, y2) = graphics.coords[idx2]
            (x, y) = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        elif graphics.type == 'circle':
            (x, y) = (graphics.x, graphics.y)
        elif graphics.type in ('rectangle', 'roundrectangle'):
            (x, y) = (graphics.x, graphics.y)
        if graphics._parent.type == 'map':
            text = graphics.name
            self.drawing.setFont(self.fontname, self.fontsize + 2)
        elif len(graphics.name) < 15:
            text = graphics.name
        else:
            text = graphics.name[:12] + '...'
        self.drawing.drawCentredString(x, y, text)
        self.drawing.setFont(self.fontname, self.fontsize)

    def __add_orthologs(self):
        if False:
            print('Hello World!')
        "Add 'ortholog' Entry elements to the drawing of the map (PRIVATE).\n\n        In KGML, these are typically line objects, so we render them\n        before the compound circles to cover the unsightly ends/junctions.\n        "
        for ortholog in self.pathway.orthologs:
            for g in ortholog.graphics:
                self.drawing.setStrokeColor(color_to_reportlab(g.fgcolor))
                self.drawing.setFillColor(color_to_reportlab(g.bgcolor))
                self.__add_graphics(g)
                if self.label_orthologs:
                    self.drawing.setFillColor(darken(g.fgcolor))
                    self.__add_labels(g)

    def __add_reaction_entries(self):
        if False:
            print('Hello World!')
        'Add Entry elements for Reactions to the map drawing (PRIVATE).\n\n        In KGML, these are typically line objects, so we render them\n        before the compound circles to cover the unsightly ends/junctions\n        '
        for reaction in self.pathway.reaction_entries:
            for g in reaction.graphics:
                self.drawing.setStrokeColor(color_to_reportlab(g.fgcolor))
                self.drawing.setFillColor(color_to_reportlab(g.bgcolor))
                self.__add_graphics(g)
                if self.label_reaction_entries:
                    self.drawing.setFillColor(darken(g.fgcolor))
                    self.__add_labels(g)

    def __add_compounds(self):
        if False:
            print('Hello World!')
        'Add compound elements to the drawing of the map (PRIVATE).'
        for compound in self.pathway.compounds:
            for g in compound.graphics:
                fillcolor = color_to_reportlab(g.bgcolor)
                if not compound.is_reactant:
                    fillcolor.alpha *= self.non_reactant_transparency
                self.drawing.setStrokeColor(color_to_reportlab(g.fgcolor))
                self.drawing.setFillColor(fillcolor)
                self.__add_graphics(g)
                if self.label_compounds:
                    if not compound.is_reactant:
                        t = 0.3
                    else:
                        t = 1
                    self.drawing.setFillColor(colors.Color(0.2, 0.2, 0.2, t))
                    self.__add_labels(g)

    def __add_genes(self):
        if False:
            while True:
                i = 10
        'Add gene elements to the drawing of the map (PRIVATE).'
        for gene in self.pathway.genes:
            for g in gene.graphics:
                self.drawing.setStrokeColor(color_to_reportlab(g.fgcolor))
                self.drawing.setFillColor(color_to_reportlab(g.bgcolor))
                self.__add_graphics(g)
                if self.label_compounds:
                    self.drawing.setFillColor(darken(g.fgcolor))
                    self.__add_labels(g)

    def __add_relations(self):
        if False:
            print('Hello World!')
        "Add relations to the map (PRIVATE).\n\n        This is tricky. There is no defined graphic in KGML for a\n        relation, and the corresponding entries are typically defined\n        as objects 'to be connected somehow'.  KEGG uses KegSketch, which\n        is not public, and most third-party software draws straight line\n        arrows, with heads to indicate the appropriate direction\n        (at both ends for reversible reactions), using solid lines for\n        ECrel relation types, and dashed lines for maplink relation types.\n\n        The relation has:\n        - entry1: 'from' node\n        - entry2: 'to' node\n        - subtype: what the relation refers to\n\n        Typically we have entry1 = map/ortholog; entry2 = map/ortholog,\n        subtype = compound.\n        "
        for relation in list(self.pathway.relations):
            if relation.type == 'maplink':
                self.drawing.setDash(6, 3)
            else:
                self.drawing.setDash()
            for s in relation.subtypes:
                subtype = self.pathway.entries[s[1]]
                self.__draw_arrow(relation.entry1, subtype)
                self.__draw_arrow(subtype, relation.entry2)

    def __draw_arrow(self, g_from, g_to):
        if False:
            for i in range(10):
                print('nop')
        'Draw an arrow between given Entry objects (PRIVATE).\n\n        Draws an arrow from the g_from Entry object to the g_to\n        Entry object; both must have Graphics objects.\n        '
        (bounds_from, bounds_to) = (g_from.bounds, g_to.bounds)
        centre_from = (0.5 * (bounds_from[0][0] + bounds_from[1][0]), 0.5 * (bounds_from[0][1] + bounds_from[1][1]))
        centre_to = (0.5 * (bounds_to[0][0] + bounds_to[1][0]), 0.5 * (bounds_to[0][1] + bounds_to[1][1]))
        p = self.drawing.beginPath()
        if bounds_to[0][0] < centre_from[0] < bounds_to[1][0]:
            if centre_to[1] > centre_from[1]:
                p.moveTo(centre_from[0], bounds_from[1][1])
                p.lineTo(centre_from[0], bounds_to[0][1])
            else:
                p.moveTo(centre_from[0], bounds_from[0][1])
                p.lineTo(centre_from[0], bounds_to[1][1])
        elif bounds_from[0][0] < centre_to[0] < bounds_from[1][0]:
            if centre_to[1] > centre_from[1]:
                p.moveTo(centre_to[0], bounds_from[1][1])
                p.lineTo(centre_to[0], bounds_to[0][1])
            else:
                p.moveTo(centre_to[0], bounds_from[0][1])
                p.lineTo(centre_to[0], bounds_to[1][1])
        self.drawing.drawPath(p)