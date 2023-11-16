from collections import OrderedDict
import logging
import urllib.parse
import numpy as np
from matplotlib import _text_helpers, dviread
from matplotlib.font_manager import FontProperties, get_font, fontManager as _fontManager
from matplotlib.ft2font import LOAD_NO_HINTING, LOAD_TARGET_LIGHT
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
_log = logging.getLogger(__name__)

class TextToPath:
    """A class that converts strings to paths."""
    FONT_SCALE = 100.0
    DPI = 72

    def __init__(self):
        if False:
            while True:
                i = 10
        self.mathtext_parser = MathTextParser('path')
        self._texmanager = None

    def _get_font(self, prop):
        if False:
            while True:
                i = 10
        '\n        Find the `FT2Font` matching font properties *prop*, with its size set.\n        '
        filenames = _fontManager._find_fonts_by_props(prop)
        font = get_font(filenames)
        font.set_size(self.FONT_SCALE, self.DPI)
        return font

    def _get_hinting_flag(self):
        if False:
            return 10
        return LOAD_NO_HINTING

    def _get_char_id(self, font, ccode):
        if False:
            print('Hello World!')
        '\n        Return a unique id for the given font and character-code set.\n        '
        return urllib.parse.quote(f'{font.postscript_name}-{ccode:x}')

    def get_text_width_height_descent(self, s, prop, ismath):
        if False:
            i = 10
            return i + 15
        fontsize = prop.get_size_in_points()
        if ismath == 'TeX':
            return TexManager().get_text_width_height_descent(s, fontsize)
        scale = fontsize / self.FONT_SCALE
        if ismath:
            prop = prop.copy()
            prop.set_size(self.FONT_SCALE)
            (width, height, descent, *_) = self.mathtext_parser.parse(s, 72, prop)
            return (width * scale, height * scale, descent * scale)
        font = self._get_font(prop)
        font.set_text(s, 0.0, flags=LOAD_NO_HINTING)
        (w, h) = font.get_width_height()
        w /= 64.0
        h /= 64.0
        d = font.get_descent()
        d /= 64.0
        return (w * scale, h * scale, d * scale)

    def get_text_path(self, prop, s, ismath=False):
        if False:
            return 10
        '\n        Convert text *s* to path (a tuple of vertices and codes for\n        matplotlib.path.Path).\n\n        Parameters\n        ----------\n        prop : `~matplotlib.font_manager.FontProperties`\n            The font properties for the text.\n        s : str\n            The text to be converted.\n        ismath : {False, True, "TeX"}\n            If True, use mathtext parser.  If "TeX", use tex for rendering.\n\n        Returns\n        -------\n        verts : list\n            A list of arrays containing the (x, y) coordinates of the vertices.\n        codes : list\n            A list of path codes.\n\n        Examples\n        --------\n        Create a list of vertices and codes from a text, and create a `.Path`\n        from those::\n\n            from matplotlib.path import Path\n            from matplotlib.text import TextToPath\n            from matplotlib.font_manager import FontProperties\n\n            fp = FontProperties(family="Humor Sans", style="italic")\n            verts, codes = TextToPath().get_text_path(fp, "ABC")\n            path = Path(verts, codes, closed=False)\n\n        Also see `TextPath` for a more direct way to create a path from a text.\n        '
        if ismath == 'TeX':
            (glyph_info, glyph_map, rects) = self.get_glyphs_tex(prop, s)
        elif not ismath:
            font = self._get_font(prop)
            (glyph_info, glyph_map, rects) = self.get_glyphs_with_font(font, s)
        else:
            (glyph_info, glyph_map, rects) = self.get_glyphs_mathtext(prop, s)
        (verts, codes) = ([], [])
        for (glyph_id, xposition, yposition, scale) in glyph_info:
            (verts1, codes1) = glyph_map[glyph_id]
            verts.extend(verts1 * scale + [xposition, yposition])
            codes.extend(codes1)
        for (verts1, codes1) in rects:
            verts.extend(verts1)
            codes.extend(codes1)
        if not verts:
            verts = np.empty((0, 2))
        return (verts, codes)

    def get_glyphs_with_font(self, font, s, glyph_map=None, return_new_glyphs_only=False):
        if False:
            print('Hello World!')
        '\n        Convert string *s* to vertices and codes using the provided ttf font.\n        '
        if glyph_map is None:
            glyph_map = OrderedDict()
        if return_new_glyphs_only:
            glyph_map_new = OrderedDict()
        else:
            glyph_map_new = glyph_map
        xpositions = []
        glyph_ids = []
        for item in _text_helpers.layout(s, font):
            char_id = self._get_char_id(item.ft_object, ord(item.char))
            glyph_ids.append(char_id)
            xpositions.append(item.x)
            if char_id not in glyph_map:
                glyph_map_new[char_id] = item.ft_object.get_path()
        ypositions = [0] * len(xpositions)
        sizes = [1.0] * len(xpositions)
        rects = []
        return (list(zip(glyph_ids, xpositions, ypositions, sizes)), glyph_map_new, rects)

    def get_glyphs_mathtext(self, prop, s, glyph_map=None, return_new_glyphs_only=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parse mathtext string *s* and convert it to a (vertices, codes) pair.\n        '
        prop = prop.copy()
        prop.set_size(self.FONT_SCALE)
        (width, height, descent, glyphs, rects) = self.mathtext_parser.parse(s, self.DPI, prop)
        if not glyph_map:
            glyph_map = OrderedDict()
        if return_new_glyphs_only:
            glyph_map_new = OrderedDict()
        else:
            glyph_map_new = glyph_map
        xpositions = []
        ypositions = []
        glyph_ids = []
        sizes = []
        for (font, fontsize, ccode, ox, oy) in glyphs:
            char_id = self._get_char_id(font, ccode)
            if char_id not in glyph_map:
                font.clear()
                font.set_size(self.FONT_SCALE, self.DPI)
                font.load_char(ccode, flags=LOAD_NO_HINTING)
                glyph_map_new[char_id] = font.get_path()
            xpositions.append(ox)
            ypositions.append(oy)
            glyph_ids.append(char_id)
            size = fontsize / self.FONT_SCALE
            sizes.append(size)
        myrects = []
        for (ox, oy, w, h) in rects:
            vert1 = [(ox, oy), (ox, oy + h), (ox + w, oy + h), (ox + w, oy), (ox, oy), (0, 0)]
            code1 = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
            myrects.append((vert1, code1))
        return (list(zip(glyph_ids, xpositions, ypositions, sizes)), glyph_map_new, myrects)

    def get_glyphs_tex(self, prop, s, glyph_map=None, return_new_glyphs_only=False):
        if False:
            i = 10
            return i + 15
        'Convert the string *s* to vertices and codes using usetex mode.'
        dvifile = TexManager().make_dvi(s, self.FONT_SCALE)
        with dviread.Dvi(dvifile, self.DPI) as dvi:
            (page,) = dvi
        if glyph_map is None:
            glyph_map = OrderedDict()
        if return_new_glyphs_only:
            glyph_map_new = OrderedDict()
        else:
            glyph_map_new = glyph_map
        (glyph_ids, xpositions, ypositions, sizes) = ([], [], [], [])
        for text in page.text:
            font = get_font(text.font_path)
            char_id = self._get_char_id(font, text.glyph)
            if char_id not in glyph_map:
                font.clear()
                font.set_size(self.FONT_SCALE, self.DPI)
                glyph_name_or_index = text.glyph_name_or_index
                if isinstance(glyph_name_or_index, str):
                    index = font.get_name_index(glyph_name_or_index)
                    font.load_glyph(index, flags=LOAD_TARGET_LIGHT)
                elif isinstance(glyph_name_or_index, int):
                    self._select_native_charmap(font)
                    font.load_char(glyph_name_or_index, flags=LOAD_TARGET_LIGHT)
                else:
                    raise TypeError(f'Glyph spec of unexpected type: {glyph_name_or_index!r}')
                glyph_map_new[char_id] = font.get_path()
            glyph_ids.append(char_id)
            xpositions.append(text.x)
            ypositions.append(text.y)
            sizes.append(text.font_size / self.FONT_SCALE)
        myrects = []
        for (ox, oy, h, w) in page.boxes:
            vert1 = [(ox, oy), (ox + w, oy), (ox + w, oy + h), (ox, oy + h), (ox, oy), (0, 0)]
            code1 = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
            myrects.append((vert1, code1))
        return (list(zip(glyph_ids, xpositions, ypositions, sizes)), glyph_map_new, myrects)

    @staticmethod
    def _select_native_charmap(font):
        if False:
            return 10
        for charmap_code in [1094992451, 1094995778]:
            try:
                font.select_charmap(charmap_code)
            except (ValueError, RuntimeError):
                pass
            else:
                break
        else:
            _log.warning('No supported encoding in font (%s).', font.fname)
text_to_path = TextToPath()

class TextPath(Path):
    """
    Create a path from the text.
    """

    def __init__(self, xy, s, size=None, prop=None, _interpolation_steps=1, usetex=False):
        if False:
            i = 10
            return i + 15
        '\n        Create a path from the text. Note that it simply is a path,\n        not an artist. You need to use the `.PathPatch` (or other artists)\n        to draw this path onto the canvas.\n\n        Parameters\n        ----------\n        xy : tuple or array of two float values\n            Position of the text. For no offset, use ``xy=(0, 0)``.\n\n        s : str\n            The text to convert to a path.\n\n        size : float, optional\n            Font size in points. Defaults to the size specified via the font\n            properties *prop*.\n\n        prop : `~matplotlib.font_manager.FontProperties`, optional\n            Font property. If not provided, will use a default\n            `.FontProperties` with parameters from the\n            :ref:`rcParams<customizing-with-dynamic-rc-settings>`.\n\n        _interpolation_steps : int, optional\n            (Currently ignored)\n\n        usetex : bool, default: False\n            Whether to use tex rendering.\n\n        Examples\n        --------\n        The following creates a path from the string "ABC" with Helvetica\n        font face; and another path from the latex fraction 1/2::\n\n            from matplotlib.text import TextPath\n            from matplotlib.font_manager import FontProperties\n\n            fp = FontProperties(family="Helvetica", style="italic")\n            path1 = TextPath((12, 12), "ABC", size=12, prop=fp)\n            path2 = TextPath((0, 0), r"$\\frac{1}{2}$", size=12, usetex=True)\n\n        Also see :doc:`/gallery/text_labels_and_annotations/demo_text_path`.\n        '
        from matplotlib.text import Text
        prop = FontProperties._from_any(prop)
        if size is None:
            size = prop.get_size_in_points()
        self._xy = xy
        self.set_size(size)
        self._cached_vertices = None
        (s, ismath) = Text(usetex=usetex)._preprocess_math(s)
        super().__init__(*text_to_path.get_text_path(prop, s, ismath=ismath), _interpolation_steps=_interpolation_steps, readonly=True)
        self._should_simplify = False

    def set_size(self, size):
        if False:
            return 10
        'Set the text size.'
        self._size = size
        self._invalid = True

    def get_size(self):
        if False:
            while True:
                i = 10
        'Get the text size.'
        return self._size

    @property
    def vertices(self):
        if False:
            print('Hello World!')
        '\n        Return the cached path after updating it if necessary.\n        '
        self._revalidate_path()
        return self._cached_vertices

    @property
    def codes(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the codes\n        '
        return self._codes

    def _revalidate_path(self):
        if False:
            return 10
        '\n        Update the path if necessary.\n\n        The path for the text is initially create with the font size of\n        `.FONT_SCALE`, and this path is rescaled to other size when necessary.\n        '
        if self._invalid or self._cached_vertices is None:
            tr = Affine2D().scale(self._size / text_to_path.FONT_SCALE).translate(*self._xy)
            self._cached_vertices = tr.transform(self._vertices)
            self._cached_vertices.flags.writeable = False
            self._invalid = False