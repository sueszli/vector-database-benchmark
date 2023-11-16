"""
Common functionality between the PDF and PS backends.
"""
from io import BytesIO
import functools
from fontTools import subset
import matplotlib as mpl
from .. import font_manager, ft2font
from .._afm import AFM
from ..backend_bases import RendererBase

@functools.lru_cache(50)
def _cached_get_afm_from_fname(fname):
    if False:
        while True:
            i = 10
    with open(fname, 'rb') as fh:
        return AFM(fh)

def get_glyphs_subset(fontfile, characters):
    if False:
        while True:
            i = 10
    '\n    Subset a TTF font\n\n    Reads the named fontfile and restricts the font to the characters.\n    Returns a serialization of the subset font as file-like object.\n\n    Parameters\n    ----------\n    fontfile : str\n        Path to the font file\n    characters : str\n        Continuous set of characters to include in subset\n    '
    options = subset.Options(glyph_names=True, recommended_glyphs=True)
    options.drop_tables += ['FFTM', 'PfEd', 'BDF', 'meta']
    if fontfile.endswith('.ttc'):
        options.font_number = 0
    with subset.load_font(fontfile, options) as font:
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(text=characters)
        subsetter.subset(font)
        fh = BytesIO()
        font.save(fh, reorderTables=False)
        return fh

class CharacterTracker:
    """
    Helper for font subsetting by the pdf and ps backends.

    Maintains a mapping of font paths to the set of character codepoints that
    are being used from that font.
    """

    def __init__(self):
        if False:
            return 10
        self.used = {}

    def track(self, font, s):
        if False:
            i = 10
            return i + 15
        'Record that string *s* is being typeset using font *font*.'
        char_to_font = font._get_fontmap(s)
        for (_c, _f) in char_to_font.items():
            self.used.setdefault(_f.fname, set()).add(ord(_c))

    def track_glyph(self, font, glyph):
        if False:
            while True:
                i = 10
        'Record that codepoint *glyph* is being typeset using font *font*.'
        self.used.setdefault(font.fname, set()).add(glyph)

class RendererPDFPSBase(RendererBase):

    def __init__(self, width, height):
        if False:
            print('Hello World!')
        super().__init__()
        self.width = width
        self.height = height

    def flipy(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def option_scale_image(self):
        if False:
            return 10
        return True

    def option_image_nocomposite(self):
        if False:
            for i in range(10):
                print('nop')
        return not mpl.rcParams['image.composite_image']

    def get_canvas_width_height(self):
        if False:
            i = 10
            return i + 15
        return (self.width * 72.0, self.height * 72.0)

    def get_text_width_height_descent(self, s, prop, ismath):
        if False:
            return 10
        if ismath == 'TeX':
            return super().get_text_width_height_descent(s, prop, ismath)
        elif ismath:
            parse = self._text2path.mathtext_parser.parse(s, 72, prop)
            return (parse.width, parse.height, parse.depth)
        elif mpl.rcParams[self._use_afm_rc_name]:
            font = self._get_font_afm(prop)
            (l, b, w, h, d) = font.get_str_bbox_and_descent(s)
            scale = prop.get_size_in_points() / 1000
            w *= scale
            h *= scale
            d *= scale
            return (w, h, d)
        else:
            font = self._get_font_ttf(prop)
            font.set_text(s, 0.0, flags=ft2font.LOAD_NO_HINTING)
            (w, h) = font.get_width_height()
            d = font.get_descent()
            scale = 1 / 64
            w *= scale
            h *= scale
            d *= scale
            return (w, h, d)

    def _get_font_afm(self, prop):
        if False:
            for i in range(10):
                print('nop')
        fname = font_manager.findfont(prop, fontext='afm', directory=self._afm_font_dir)
        return _cached_get_afm_from_fname(fname)

    def _get_font_ttf(self, prop):
        if False:
            return 10
        fnames = font_manager.fontManager._find_fonts_by_props(prop)
        font = font_manager.get_font(fnames)
        font.clear()
        font.set_size(prop.get_size_in_points(), 72)
        return font