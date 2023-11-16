"""
A PostScript backend, which can produce both PostScript .ps and .eps.
"""
import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import math
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, RendererBase
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps
_log = logging.getLogger(__name__)
debugPS = False

@_api.deprecated('3.7')
class PsBackendHelper:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._cached = {}

@_api.caching_module_getattr
class __getattr__:
    ps_backend_helper = _api.deprecated('3.7', obj_type='')(property(lambda self: PsBackendHelper()))
    psDefs = _api.deprecated('3.8', obj_type='')(property(lambda self: _psDefs))
papersize = {'letter': (8.5, 11), 'legal': (8.5, 14), 'ledger': (11, 17), 'a0': (33.11, 46.81), 'a1': (23.39, 33.11), 'a2': (16.54, 23.39), 'a3': (11.69, 16.54), 'a4': (8.27, 11.69), 'a5': (5.83, 8.27), 'a6': (4.13, 5.83), 'a7': (2.91, 4.13), 'a8': (2.05, 2.91), 'a9': (1.46, 2.05), 'a10': (1.02, 1.46), 'b0': (40.55, 57.32), 'b1': (28.66, 40.55), 'b2': (20.27, 28.66), 'b3': (14.33, 20.27), 'b4': (10.11, 14.33), 'b5': (7.16, 10.11), 'b6': (5.04, 7.16), 'b7': (3.58, 5.04), 'b8': (2.51, 3.58), 'b9': (1.76, 2.51), 'b10': (1.26, 1.76)}

def _get_papertype(w, h):
    if False:
        print('Hello World!')
    for (key, (pw, ph)) in sorted(papersize.items(), reverse=True):
        if key.startswith('l'):
            continue
        if w < pw and h < ph:
            return key
    return 'a0'

def _nums_to_str(*args, sep=' '):
    if False:
        for i in range(10):
            print('nop')
    return sep.join((f'{arg:1.3f}'.rstrip('0').rstrip('.') for arg in args))

def _move_path_to_path_or_stream(src, dst):
    if False:
        while True:
            i = 10
    '\n    Move the contents of file at *src* to path-or-filelike *dst*.\n\n    If *dst* is a path, the metadata of *src* are *not* copied.\n    '
    if is_writable_file_like(dst):
        fh = open(src, encoding='latin-1') if file_requires_unicode(dst) else open(src, 'rb')
        with fh:
            shutil.copyfileobj(fh, dst)
    else:
        shutil.move(src, dst, copy_function=shutil.copyfile)

def _font_to_ps_type3(font_path, chars):
    if False:
        return 10
    '\n    Subset *chars* from the font at *font_path* into a Type 3 font.\n\n    Parameters\n    ----------\n    font_path : path-like\n        Path to the font to be subsetted.\n    chars : str\n        The characters to include in the subsetted font.\n\n    Returns\n    -------\n    str\n        The string representation of a Type 3 font, which can be included\n        verbatim into a PostScript file.\n    '
    font = get_font(font_path, hinting_factor=1)
    glyph_ids = [font.get_char_index(c) for c in chars]
    preamble = '%!PS-Adobe-3.0 Resource-Font\n%%Creator: Converted from TrueType to Type 3 by Matplotlib.\n10 dict begin\n/FontName /{font_name} def\n/PaintType 0 def\n/FontMatrix [{inv_units_per_em} 0 0 {inv_units_per_em} 0 0] def\n/FontBBox [{bbox}] def\n/FontType 3 def\n/Encoding [{encoding}] def\n/CharStrings {num_glyphs} dict dup begin\n/.notdef 0 def\n'.format(font_name=font.postscript_name, inv_units_per_em=1 / font.units_per_EM, bbox=' '.join(map(str, font.bbox)), encoding=' '.join((f'/{font.get_glyph_name(glyph_id)}' for glyph_id in glyph_ids)), num_glyphs=len(glyph_ids) + 1)
    postamble = '\nend readonly def\n\n/BuildGlyph {\n exch begin\n CharStrings exch\n 2 copy known not {pop /.notdef} if\n true 3 1 roll get exec\n end\n} _d\n\n/BuildChar {\n 1 index /Encoding get exch get\n 1 index /BuildGlyph get exec\n} _d\n\nFontName currentdict end definefont pop\n'
    entries = []
    for glyph_id in glyph_ids:
        g = font.load_glyph(glyph_id, LOAD_NO_SCALE)
        (v, c) = font.get_path()
        entries.append('/%(name)s{%(bbox)s sc\n' % {'name': font.get_glyph_name(glyph_id), 'bbox': ' '.join(map(str, [g.horiAdvance, 0, *g.bbox]))} + _path.convert_to_string(Path(v * 64, c), None, None, False, None, 0, [b'm', b'l', b'', b'c', b''], True).decode('ascii') + 'ce} _d')
    return preamble + '\n'.join(entries) + postamble

def _font_to_ps_type42(font_path, chars, fh):
    if False:
        for i in range(10):
            print('nop')
    '\n    Subset *chars* from the font at *font_path* into a Type 42 font at *fh*.\n\n    Parameters\n    ----------\n    font_path : path-like\n        Path to the font to be subsetted.\n    chars : str\n        The characters to include in the subsetted font.\n    fh : file-like\n        Where to write the font.\n    '
    subset_str = ''.join((chr(c) for c in chars))
    _log.debug('SUBSET %s characters: %s', font_path, subset_str)
    try:
        fontdata = _backend_pdf_ps.get_glyphs_subset(font_path, subset_str)
        _log.debug('SUBSET %s %d -> %d', font_path, os.stat(font_path).st_size, fontdata.getbuffer().nbytes)
        font = FT2Font(fontdata)
        glyph_ids = [font.get_char_index(c) for c in chars]
        with TemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, 'tmp.ttf')
            with open(tmpfile, 'wb') as tmp:
                tmp.write(fontdata.getvalue())
            convert_ttf_to_ps(os.fsencode(tmpfile), fh, 42, glyph_ids)
    except RuntimeError:
        _log.warning('The PostScript backend does not currently support the selected font.')
        raise

def _log_if_debug_on(meth):
    if False:
        while True:
            i = 10
    '\n    Wrap `RendererPS` method *meth* to emit a PS comment with the method name,\n    if the global flag `debugPS` is set.\n    '

    @functools.wraps(meth)
    def wrapper(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if debugPS:
            self._pswriter.write(f'% {meth.__name__}\n')
        return meth(self, *args, **kwargs)
    return wrapper

class RendererPS(_backend_pdf_ps.RendererPDFPSBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles.
    """
    _afm_font_dir = cbook._get_data_path('fonts/afm')
    _use_afm_rc_name = 'ps.useafm'

    def __init__(self, width, height, pswriter, imagedpi=72):
        if False:
            return 10
        super().__init__(width, height)
        self._pswriter = pswriter
        if mpl.rcParams['text.usetex']:
            self.textcnt = 0
            self.psfrag = []
        self.imagedpi = imagedpi
        self.color = None
        self.linewidth = None
        self.linejoin = None
        self.linecap = None
        self.linedash = None
        self.fontname = None
        self.fontsize = None
        self._hatches = {}
        self.image_magnification = imagedpi / 72
        self._clip_paths = {}
        self._path_collection_id = 0
        self._character_tracker = _backend_pdf_ps.CharacterTracker()
        self._logwarn_once = functools.cache(_log.warning)

    def _is_transparent(self, rgb_or_rgba):
        if False:
            while True:
                i = 10
        if rgb_or_rgba is None:
            return True
        elif len(rgb_or_rgba) == 4:
            if rgb_or_rgba[3] == 0:
                return True
            if rgb_or_rgba[3] != 1:
                self._logwarn_once('The PostScript backend does not support transparency; partially transparent artists will be rendered opaque.')
            return False
        else:
            return False

    def set_color(self, r, g, b, store=True):
        if False:
            return 10
        if (r, g, b) != self.color:
            self._pswriter.write(f'{_nums_to_str(r)} setgray\n' if r == g == b else f'{_nums_to_str(r, g, b)} setrgbcolor\n')
            if store:
                self.color = (r, g, b)

    def set_linewidth(self, linewidth, store=True):
        if False:
            print('Hello World!')
        linewidth = float(linewidth)
        if linewidth != self.linewidth:
            self._pswriter.write(f'{_nums_to_str(linewidth)} setlinewidth\n')
            if store:
                self.linewidth = linewidth

    @staticmethod
    def _linejoin_cmd(linejoin):
        if False:
            i = 10
            return i + 15
        linejoin = {'miter': 0, 'round': 1, 'bevel': 2, 0: 0, 1: 1, 2: 2}[linejoin]
        return f'{linejoin:d} setlinejoin\n'

    def set_linejoin(self, linejoin, store=True):
        if False:
            i = 10
            return i + 15
        if linejoin != self.linejoin:
            self._pswriter.write(self._linejoin_cmd(linejoin))
            if store:
                self.linejoin = linejoin

    @staticmethod
    def _linecap_cmd(linecap):
        if False:
            i = 10
            return i + 15
        linecap = {'butt': 0, 'round': 1, 'projecting': 2, 0: 0, 1: 1, 2: 2}[linecap]
        return f'{linecap:d} setlinecap\n'

    def set_linecap(self, linecap, store=True):
        if False:
            i = 10
            return i + 15
        if linecap != self.linecap:
            self._pswriter.write(self._linecap_cmd(linecap))
            if store:
                self.linecap = linecap

    def set_linedash(self, offset, seq, store=True):
        if False:
            for i in range(10):
                print('nop')
        if self.linedash is not None:
            (oldo, oldseq) = self.linedash
            if np.array_equal(seq, oldseq) and oldo == offset:
                return
        self._pswriter.write(f'[{_nums_to_str(*seq)}] {_nums_to_str(offset)} setdash\n' if seq is not None and len(seq) else '[] 0 setdash\n')
        if store:
            self.linedash = (offset, seq)

    def set_font(self, fontname, fontsize, store=True):
        if False:
            for i in range(10):
                print('nop')
        if (fontname, fontsize) != (self.fontname, self.fontsize):
            self._pswriter.write(f'/{fontname} {fontsize:1.3f} selectfont\n')
            if store:
                self.fontname = fontname
                self.fontsize = fontsize

    def create_hatch(self, hatch):
        if False:
            for i in range(10):
                print('nop')
        sidelen = 72
        if hatch in self._hatches:
            return self._hatches[hatch]
        name = 'H%d' % len(self._hatches)
        linewidth = mpl.rcParams['hatch.linewidth']
        pageheight = self.height * 72
        self._pswriter.write(f'  << /PatternType 1\n     /PaintType 2\n     /TilingType 2\n     /BBox[0 0 {sidelen:d} {sidelen:d}]\n     /XStep {sidelen:d}\n     /YStep {sidelen:d}\n\n     /PaintProc {{\n        pop\n        {linewidth:g} setlinewidth\n{self._convert_path(Path.hatch(hatch), Affine2D().scale(sidelen), simplify=False)}\n        gsave\n        fill\n        grestore\n        stroke\n     }} bind\n   >>\n   matrix\n   0 {pageheight:g} translate\n   makepattern\n   /{name} exch def\n')
        self._hatches[hatch] = name
        return name

    def get_image_magnification(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the factor by which to magnify images passed to draw_image.\n        Allows a backend to have images at a different resolution to other\n        artists.\n        '
        return self.image_magnification

    def _convert_path(self, path, transform, clip=False, simplify=None):
        if False:
            while True:
                i = 10
        if clip:
            clip = (0.0, 0.0, self.width * 72.0, self.height * 72.0)
        else:
            clip = None
        return _path.convert_to_string(path, transform, clip, simplify, None, 6, [b'm', b'l', b'', b'c', b'cl'], True).decode('ascii')

    def _get_clip_cmd(self, gc):
        if False:
            return 10
        clip = []
        rect = gc.get_clip_rectangle()
        if rect is not None:
            clip.append(f'{_nums_to_str(*rect.p0, *rect.size)} rectclip\n')
        (path, trf) = gc.get_clip_path()
        if path is not None:
            key = (path, id(trf))
            custom_clip_cmd = self._clip_paths.get(key)
            if custom_clip_cmd is None:
                custom_clip_cmd = 'c%d' % len(self._clip_paths)
                self._pswriter.write(f'/{custom_clip_cmd} {{\n{self._convert_path(path, trf, simplify=False)}\nclip\nnewpath\n}} bind def\n')
                self._clip_paths[key] = custom_clip_cmd
            clip.append(f'{custom_clip_cmd}\n')
        return ''.join(clip)

    @_log_if_debug_on
    def draw_image(self, gc, x, y, im, transform=None):
        if False:
            print('Hello World!')
        (h, w) = im.shape[:2]
        imagecmd = 'false 3 colorimage'
        data = im[::-1, :, :3]
        hexdata = data.tobytes().hex('\n', -64)
        if transform is None:
            matrix = '1 0 0 1 0 0'
            xscale = w / self.image_magnification
            yscale = h / self.image_magnification
        else:
            matrix = ' '.join(map(str, transform.frozen().to_values()))
            xscale = 1.0
            yscale = 1.0
        self._pswriter.write(f'gsave\n{self._get_clip_cmd(gc)}\n{x:g} {y:g} translate\n[{matrix}] concat\n{xscale:g} {yscale:g} scale\n/DataString {w:d} string def\n{w:d} {h:d} 8 [ {w:d} 0 0 -{h:d} 0 {h:d} ]\n{{\ncurrentfile DataString readhexstring pop\n}} bind {imagecmd}\n{hexdata}\ngrestore\n')

    @_log_if_debug_on
    def draw_path(self, gc, path, transform, rgbFace=None):
        if False:
            i = 10
            return i + 15
        clip = rgbFace is None and gc.get_hatch_path() is None
        simplify = path.should_simplify and clip
        ps = self._convert_path(path, transform, clip=clip, simplify=simplify)
        self._draw_ps(ps, gc, rgbFace)

    @_log_if_debug_on
    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        if False:
            return 10
        ps_color = None if self._is_transparent(rgbFace) else f'{_nums_to_str(rgbFace[0])} setgray' if rgbFace[0] == rgbFace[1] == rgbFace[2] else f'{_nums_to_str(*rgbFace[:3])} setrgbcolor'
        ps_cmd = ['/o {', 'gsave', 'newpath', 'translate']
        lw = gc.get_linewidth()
        alpha = gc.get_alpha() if gc.get_forced_alpha() or len(gc.get_rgb()) == 3 else gc.get_rgb()[3]
        stroke = lw > 0 and alpha > 0
        if stroke:
            ps_cmd.append('%.1f setlinewidth' % lw)
            ps_cmd.append(self._linejoin_cmd(gc.get_joinstyle()))
            ps_cmd.append(self._linecap_cmd(gc.get_capstyle()))
        ps_cmd.append(self._convert_path(marker_path, marker_trans, simplify=False))
        if rgbFace:
            if stroke:
                ps_cmd.append('gsave')
            if ps_color:
                ps_cmd.extend([ps_color, 'fill'])
            if stroke:
                ps_cmd.append('grestore')
        if stroke:
            ps_cmd.append('stroke')
        ps_cmd.extend(['grestore', '} bind def'])
        for (vertices, code) in path.iter_segments(trans, clip=(0, 0, self.width * 72, self.height * 72), simplify=False):
            if len(vertices):
                (x, y) = vertices[-2:]
                ps_cmd.append(f'{x:g} {y:g} o')
        ps = '\n'.join(ps_cmd)
        self._draw_ps(ps, gc, rgbFace, fill=False, stroke=False)

    @_log_if_debug_on
    def draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position):
        if False:
            return 10
        len_path = len(paths[0].vertices) if len(paths) > 0 else 0
        uses_per_path = self._iter_collection_uses_per_path(paths, all_transforms, offsets, facecolors, edgecolors)
        should_do_optimization = len_path + 3 * uses_per_path + 3 < (len_path + 2) * uses_per_path
        if not should_do_optimization:
            return RendererBase.draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position)
        path_codes = []
        for (i, (path, transform)) in enumerate(self._iter_collection_raw_paths(master_transform, paths, all_transforms)):
            name = 'p%d_%d' % (self._path_collection_id, i)
            path_bytes = self._convert_path(path, transform, simplify=False)
            self._pswriter.write(f'/{name} {{\nnewpath\ntranslate\n{path_bytes}\n}} bind def\n')
            path_codes.append(name)
        for (xo, yo, path_id, gc0, rgbFace) in self._iter_collection(gc, path_codes, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position):
            ps = f'{xo:g} {yo:g} {path_id}'
            self._draw_ps(ps, gc0, rgbFace)
        self._path_collection_id += 1

    @_log_if_debug_on
    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        if False:
            return 10
        if self._is_transparent(gc.get_rgb()):
            return
        if not hasattr(self, 'psfrag'):
            self._logwarn_once("The PS backend determines usetex status solely based on rcParams['text.usetex'] and does not support having usetex=True only for some elements; this element will thus be rendered as if usetex=False.")
            self.draw_text(gc, x, y, s, prop, angle, False, mtext)
            return
        (w, h, bl) = self.get_text_width_height_descent(s, prop, ismath='TeX')
        fontsize = prop.get_size_in_points()
        thetext = 'psmarker%d' % self.textcnt
        color = _nums_to_str(*gc.get_rgb()[:3], sep=',')
        fontcmd = {'sans-serif': '{\\sffamily %s}', 'monospace': '{\\ttfamily %s}'}.get(mpl.rcParams['font.family'][0], '{\\rmfamily %s}')
        s = fontcmd % s
        tex = '\\color[rgb]{%s} %s' % (color, s)
        rangle = np.radians(angle + 90)
        pos = _nums_to_str(x - bl * np.cos(rangle), y - bl * np.sin(rangle))
        self.psfrag.append('\\psfrag{%s}[bl][bl][1][%f]{\\fontsize{%f}{%f}%s}' % (thetext, angle, fontsize, fontsize * 1.25, tex))
        self._pswriter.write(f'gsave\n{pos} moveto\n({thetext})\nshow\ngrestore\n')
        self.textcnt += 1

    @_log_if_debug_on
    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if False:
            while True:
                i = 10
        if self._is_transparent(gc.get_rgb()):
            return
        if ismath == 'TeX':
            return self.draw_tex(gc, x, y, s, prop, angle)
        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)
        stream = []
        if mpl.rcParams['ps.useafm']:
            font = self._get_font_afm(prop)
            ps_name = font.postscript_name.encode('ascii', 'replace').decode('ascii')
            scale = 0.001 * prop.get_size_in_points()
            thisx = 0
            last_name = None
            for c in s:
                name = uni2type1.get(ord(c), f'uni{ord(c):04X}')
                try:
                    width = font.get_width_from_char_name(name)
                except KeyError:
                    name = 'question'
                    width = font.get_width_char('?')
                kern = font.get_kern_dist_from_name(last_name, name)
                last_name = name
                thisx += kern * scale
                stream.append((ps_name, thisx, name))
                thisx += width * scale
        else:
            font = self._get_font_ttf(prop)
            self._character_tracker.track(font, s)
            for item in _text_helpers.layout(s, font):
                ps_name = item.ft_object.postscript_name.encode('ascii', 'replace').decode('ascii')
                glyph_name = item.ft_object.get_glyph_name(item.glyph_idx)
                stream.append((ps_name, item.x, glyph_name))
        self.set_color(*gc.get_rgb())
        for (ps_name, group) in itertools.groupby(stream, lambda entry: entry[0]):
            self.set_font(ps_name, prop.get_size_in_points(), False)
            thetext = '\n'.join((f'{x:g} 0 m /{name:s} glyphshow' for (_, x, name) in group))
            self._pswriter.write(f'gsave\n{self._get_clip_cmd(gc)}\n{x:g} {y:g} translate\n{angle:g} rotate\n{thetext}\ngrestore\n')

    @_log_if_debug_on
    def draw_mathtext(self, gc, x, y, s, prop, angle):
        if False:
            while True:
                i = 10
        'Draw the math text using matplotlib.mathtext.'
        (width, height, descent, glyphs, rects) = self._text2path.mathtext_parser.parse(s, 72, prop)
        self.set_color(*gc.get_rgb())
        self._pswriter.write(f'gsave\n{x:g} {y:g} translate\n{angle:g} rotate\n')
        lastfont = None
        for (font, fontsize, num, ox, oy) in glyphs:
            self._character_tracker.track_glyph(font, num)
            if (font.postscript_name, fontsize) != lastfont:
                lastfont = (font.postscript_name, fontsize)
                self._pswriter.write(f'/{font.postscript_name} {fontsize} selectfont\n')
            glyph_name = font.get_name_char(chr(num)) if isinstance(font, AFM) else font.get_glyph_name(font.get_char_index(num))
            self._pswriter.write(f'{ox:g} {oy:g} moveto\n/{glyph_name} glyphshow\n')
        for (ox, oy, w, h) in rects:
            self._pswriter.write(f'{ox} {oy} {w} {h} rectfill\n')
        self._pswriter.write('grestore\n')

    @_log_if_debug_on
    def draw_gouraud_triangles(self, gc, points, colors, trans):
        if False:
            for i in range(10):
                print('nop')
        assert len(points) == len(colors)
        if len(points) == 0:
            return
        assert points.ndim == 3
        assert points.shape[1] == 3
        assert points.shape[2] == 2
        assert colors.ndim == 3
        assert colors.shape[1] == 3
        assert colors.shape[2] == 4
        shape = points.shape
        flat_points = points.reshape((shape[0] * shape[1], 2))
        flat_points = trans.transform(flat_points)
        flat_colors = colors.reshape((shape[0] * shape[1], 4))
        points_min = np.min(flat_points, axis=0) - (1 << 12)
        points_max = np.max(flat_points, axis=0) + (1 << 12)
        factor = np.ceil((2 ** 32 - 1) / (points_max - points_min))
        (xmin, ymin) = points_min
        (xmax, ymax) = points_max
        data = np.empty(shape[0] * shape[1], dtype=[('flags', 'u1'), ('points', '2>u4'), ('colors', '3u1')])
        data['flags'] = 0
        data['points'] = (flat_points - points_min) * factor
        data['colors'] = flat_colors[:, :3] * 255.0
        hexdata = data.tobytes().hex('\n', -64)
        self._pswriter.write(f'gsave\n<< /ShadingType 4\n   /ColorSpace [/DeviceRGB]\n   /BitsPerCoordinate 32\n   /BitsPerComponent 8\n   /BitsPerFlag 8\n   /AntiAlias true\n   /Decode [ {xmin:g} {xmax:g} {ymin:g} {ymax:g} 0 1 0 1 0 1 ]\n   /DataSource <\n{hexdata}\n>\n>>\nshfill\ngrestore\n')

    def _draw_ps(self, ps, gc, rgbFace, *, fill=True, stroke=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Emit the PostScript snippet *ps* with all the attributes from *gc*\n        applied.  *ps* must consist of PostScript commands to construct a path.\n\n        The *fill* and/or *stroke* kwargs can be set to False if the *ps*\n        string already includes filling and/or stroking, in which case\n        `_draw_ps` is just supplying properties and clipping.\n        '
        write = self._pswriter.write
        mightstroke = gc.get_linewidth() > 0 and (not self._is_transparent(gc.get_rgb()))
        if not mightstroke:
            stroke = False
        if self._is_transparent(rgbFace):
            fill = False
        hatch = gc.get_hatch()
        if mightstroke:
            self.set_linewidth(gc.get_linewidth())
            self.set_linejoin(gc.get_joinstyle())
            self.set_linecap(gc.get_capstyle())
            self.set_linedash(*gc.get_dashes())
        if mightstroke or hatch:
            self.set_color(*gc.get_rgb()[:3])
        write('gsave\n')
        write(self._get_clip_cmd(gc))
        write(ps.strip())
        write('\n')
        if fill:
            if stroke or hatch:
                write('gsave\n')
            self.set_color(*rgbFace[:3], store=False)
            write('fill\n')
            if stroke or hatch:
                write('grestore\n')
        if hatch:
            hatch_name = self.create_hatch(hatch)
            write('gsave\n')
            write(_nums_to_str(*gc.get_hatch_color()[:3]))
            write(f' {hatch_name} setpattern fill grestore\n')
        if stroke:
            write('stroke\n')
        write('grestore\n')

class _Orientation(Enum):
    (portrait, landscape) = range(2)

    def swap_if_landscape(self, shape):
        if False:
            i = 10
            return i + 15
        return shape[::-1] if self.name == 'landscape' else shape

class FigureCanvasPS(FigureCanvasBase):
    fixed_dpi = 72
    filetypes = {'ps': 'Postscript', 'eps': 'Encapsulated Postscript'}

    def get_default_filetype(self):
        if False:
            i = 10
            return i + 15
        return 'ps'

    def _print_ps(self, fmt, outfile, *, metadata=None, papertype=None, orientation='portrait', bbox_inches_restore=None, **kwargs):
        if False:
            print('Hello World!')
        dpi = self.figure.dpi
        self.figure.dpi = 72
        dsc_comments = {}
        if isinstance(outfile, (str, os.PathLike)):
            filename = pathlib.Path(outfile).name
            dsc_comments['Title'] = filename.encode('ascii', 'replace').decode('ascii')
        dsc_comments['Creator'] = (metadata or {}).get('Creator', f'Matplotlib v{mpl.__version__}, https://matplotlib.org/')
        source_date_epoch = os.getenv('SOURCE_DATE_EPOCH')
        dsc_comments['CreationDate'] = datetime.datetime.fromtimestamp(int(source_date_epoch), datetime.timezone.utc).strftime('%a %b %d %H:%M:%S %Y') if source_date_epoch else time.ctime()
        dsc_comments = '\n'.join((f'%%{k}: {v}' for (k, v) in dsc_comments.items()))
        if papertype is None:
            papertype = mpl.rcParams['ps.papersize']
        papertype = papertype.lower()
        _api.check_in_list(['figure', 'auto', *papersize], papertype=papertype)
        orientation = _api.check_getitem(_Orientation, orientation=orientation.lower())
        printer = self._print_figure_tex if mpl.rcParams['text.usetex'] else self._print_figure
        printer(fmt, outfile, dpi=dpi, dsc_comments=dsc_comments, orientation=orientation, papertype=papertype, bbox_inches_restore=bbox_inches_restore, **kwargs)

    def _print_figure(self, fmt, outfile, *, dpi, dsc_comments, orientation, papertype, bbox_inches_restore=None):
        if False:
            return 10
        '\n        Render the figure to a filesystem path or a file-like object.\n\n        Parameters are as for `.print_figure`, except that *dsc_comments* is a\n        string containing Document Structuring Convention comments,\n        generated from the *metadata* parameter to `.print_figure`.\n        '
        is_eps = fmt == 'eps'
        if not (isinstance(outfile, (str, os.PathLike)) or is_writable_file_like(outfile)):
            raise ValueError('outfile must be a path or a file-like object')
        (width, height) = self.figure.get_size_inches()
        if papertype == 'auto':
            papertype = _get_papertype(*orientation.swap_if_landscape((width, height)))
        if is_eps or papertype == 'figure':
            (paper_width, paper_height) = (width, height)
        else:
            (paper_width, paper_height) = orientation.swap_if_landscape(papersize[papertype])
        xo = 72 * 0.5 * (paper_width - width)
        yo = 72 * 0.5 * (paper_height - height)
        llx = xo
        lly = yo
        urx = llx + self.figure.bbox.width
        ury = lly + self.figure.bbox.height
        rotation = 0
        if orientation is _Orientation.landscape:
            (llx, lly, urx, ury) = (lly, llx, ury, urx)
            (xo, yo) = (72 * paper_height - yo, xo)
            rotation = 90
        bbox = (llx, lly, urx, ury)
        self._pswriter = StringIO()
        ps_renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)
        renderer = MixedModeRenderer(self.figure, width, height, dpi, ps_renderer, bbox_inches_restore=bbox_inches_restore)
        self.figure.draw(renderer)

        def print_figure_impl(fh):
            if False:
                return 10
            if is_eps:
                print('%!PS-Adobe-3.0 EPSF-3.0', file=fh)
            else:
                print('%!PS-Adobe-3.0', file=fh)
                if papertype != 'figure':
                    print(f'%%DocumentPaperSizes: {papertype}', file=fh)
                print('%%Pages: 1', file=fh)
            print(f'%%LanguageLevel: 3\n{dsc_comments}\n%%Orientation: {orientation.name}\n{_get_bbox_header(bbox)}\n%%EndComments\n', end='', file=fh)
            Ndict = len(_psDefs)
            print('%%BeginProlog', file=fh)
            if not mpl.rcParams['ps.useafm']:
                Ndict += len(ps_renderer._character_tracker.used)
            print('/mpldict %d dict def' % Ndict, file=fh)
            print('mpldict begin', file=fh)
            print('\n'.join(_psDefs), file=fh)
            if not mpl.rcParams['ps.useafm']:
                for (font_path, chars) in ps_renderer._character_tracker.used.items():
                    if not chars:
                        continue
                    fonttype = mpl.rcParams['ps.fonttype']
                    if len(chars) > 255:
                        fonttype = 42
                    fh.flush()
                    if fonttype == 3:
                        fh.write(_font_to_ps_type3(font_path, chars))
                    else:
                        _font_to_ps_type42(font_path, chars, fh)
            print('end', file=fh)
            print('%%EndProlog', file=fh)
            if not is_eps:
                print('%%Page: 1 1', file=fh)
            print('mpldict begin', file=fh)
            print('%s translate' % _nums_to_str(xo, yo), file=fh)
            if rotation:
                print('%d rotate' % rotation, file=fh)
            print(f'0 0 {_nums_to_str(width * 72, height * 72)} rectclip', file=fh)
            print(self._pswriter.getvalue(), file=fh)
            print('end', file=fh)
            print('showpage', file=fh)
            if not is_eps:
                print('%%EOF', file=fh)
            fh.flush()
        if mpl.rcParams['ps.usedistiller']:
            with TemporaryDirectory() as tmpdir:
                tmpfile = os.path.join(tmpdir, 'tmp.ps')
                with open(tmpfile, 'w', encoding='latin-1') as fh:
                    print_figure_impl(fh)
                if mpl.rcParams['ps.usedistiller'] == 'ghostscript':
                    _try_distill(gs_distill, tmpfile, is_eps, ptype=papertype, bbox=bbox)
                elif mpl.rcParams['ps.usedistiller'] == 'xpdf':
                    _try_distill(xpdf_distill, tmpfile, is_eps, ptype=papertype, bbox=bbox)
                _move_path_to_path_or_stream(tmpfile, outfile)
        else:
            with cbook.open_file_cm(outfile, 'w', encoding='latin-1') as file:
                if not file_requires_unicode(file):
                    file = codecs.getwriter('latin-1')(file)
                print_figure_impl(file)

    def _print_figure_tex(self, fmt, outfile, *, dpi, dsc_comments, orientation, papertype, bbox_inches_restore=None):
        if False:
            return 10
        '\n        If :rc:`text.usetex` is True, a temporary pair of tex/eps files\n        are created to allow tex to manage the text layout via the PSFrags\n        package. These files are processed to yield the final ps or eps file.\n\n        The rest of the behavior is as for `._print_figure`.\n        '
        is_eps = fmt == 'eps'
        (width, height) = self.figure.get_size_inches()
        xo = 0
        yo = 0
        llx = xo
        lly = yo
        urx = llx + self.figure.bbox.width
        ury = lly + self.figure.bbox.height
        bbox = (llx, lly, urx, ury)
        self._pswriter = StringIO()
        ps_renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)
        renderer = MixedModeRenderer(self.figure, width, height, dpi, ps_renderer, bbox_inches_restore=bbox_inches_restore)
        self.figure.draw(renderer)
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir, 'tmp.ps')
            tmppath.write_text(f"%!PS-Adobe-3.0 EPSF-3.0\n%%LanguageLevel: 3\n{dsc_comments}\n{_get_bbox_header(bbox)}\n%%EndComments\n%%BeginProlog\n/mpldict {len(_psDefs)} dict def\nmpldict begin\n{''.join(_psDefs)}\nend\n%%EndProlog\nmpldict begin\n{_nums_to_str(xo, yo)} translate\n0 0 {_nums_to_str(width * 72, height * 72)} rectclip\n{self._pswriter.getvalue()}\nend\nshowpage\n", encoding='latin-1')
            if orientation is _Orientation.landscape:
                (width, height) = (height, width)
                bbox = (lly, llx, ury, urx)
            if is_eps or papertype == 'figure':
                (paper_width, paper_height) = orientation.swap_if_landscape(self.figure.get_size_inches())
            else:
                if papertype == 'auto':
                    papertype = _get_papertype(width, height)
                (paper_width, paper_height) = papersize[papertype]
            psfrag_rotated = _convert_psfrags(tmppath, ps_renderer.psfrag, paper_width, paper_height, orientation.name)
            if mpl.rcParams['ps.usedistiller'] == 'ghostscript' or mpl.rcParams['text.usetex']:
                _try_distill(gs_distill, tmppath, is_eps, ptype=papertype, bbox=bbox, rotated=psfrag_rotated)
            elif mpl.rcParams['ps.usedistiller'] == 'xpdf':
                _try_distill(xpdf_distill, tmppath, is_eps, ptype=papertype, bbox=bbox, rotated=psfrag_rotated)
            _move_path_to_path_or_stream(tmppath, outfile)
    print_ps = functools.partialmethod(_print_ps, 'ps')
    print_eps = functools.partialmethod(_print_ps, 'eps')

    def draw(self):
        if False:
            for i in range(10):
                print('nop')
        self.figure.draw_without_rendering()
        return super().draw()

def _convert_psfrags(tmppath, psfrags, paper_width, paper_height, orientation):
    if False:
        for i in range(10):
            print('nop')
    '\n    When we want to use the LaTeX backend with postscript, we write PSFrag tags\n    to a temporary postscript file, each one marking a position for LaTeX to\n    render some text. convert_psfrags generates a LaTeX document containing the\n    commands to convert those tags to text. LaTeX/dvips produces the postscript\n    file that includes the actual text.\n    '
    with mpl.rc_context({'text.latex.preamble': mpl.rcParams['text.latex.preamble'] + mpl.texmanager._usepackage_if_not_loaded('color') + mpl.texmanager._usepackage_if_not_loaded('graphicx') + mpl.texmanager._usepackage_if_not_loaded('psfrag') + '\\geometry{papersize={%(width)sin,%(height)sin},margin=0in}' % {'width': paper_width, 'height': paper_height}}):
        dvifile = TexManager().make_dvi('\n\\begin{figure}\n  \\centering\\leavevmode\n  %(psfrags)s\n  \\includegraphics*[angle=%(angle)s]{%(epsfile)s}\n\\end{figure}' % {'psfrags': '\n'.join(psfrags), 'angle': 90 if orientation == 'landscape' else 0, 'epsfile': tmppath.resolve().as_posix()}, fontsize=10)
    with TemporaryDirectory() as tmpdir:
        psfile = os.path.join(tmpdir, 'tmp.ps')
        cbook._check_and_log_subprocess(['dvips', '-q', '-R0', '-o', psfile, dvifile], _log)
        shutil.move(psfile, tmppath)
    with open(tmppath) as fh:
        psfrag_rotated = 'Landscape' in fh.read(1000)
    return psfrag_rotated

def _try_distill(func, tmppath, *args, **kwargs):
    if False:
        print('Hello World!')
    try:
        func(str(tmppath), *args, **kwargs)
    except mpl.ExecutableNotFoundError as exc:
        _log.warning('%s.  Distillation step skipped.', exc)

def gs_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
    if False:
        i = 10
        return i + 15
    "\n    Use ghostscript's pswrite or epswrite device to distill a file.\n    This yields smaller files without illegal encapsulated postscript\n    operators. The output is low-level, converting text to outlines.\n    "
    if eps:
        paper_option = ['-dEPSCrop']
    elif ptype == 'figure':
        paper_option = [f'-dDEVICEWIDTHPOINTS={bbox[2]}', f'-dDEVICEHEIGHTPOINTS={bbox[3]}']
    else:
        paper_option = [f'-sPAPERSIZE={ptype}']
    psfile = tmpfile + '.ps'
    dpi = mpl.rcParams['ps.distiller.res']
    cbook._check_and_log_subprocess([mpl._get_executable_info('gs').executable, '-dBATCH', '-dNOPAUSE', '-r%d' % dpi, '-sDEVICE=ps2write', *paper_option, f'-sOutputFile={psfile}', tmpfile], _log)
    os.remove(tmpfile)
    shutil.move(psfile, tmpfile)
    if eps:
        pstoeps(tmpfile, bbox, rotated=rotated)

def xpdf_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
    if False:
        return 10
    "\n    Use ghostscript's ps2pdf and xpdf's/poppler's pdftops to distill a file.\n    This yields smaller files without illegal encapsulated postscript\n    operators. This distiller is preferred, generating high-level postscript\n    output that treats text as text.\n    "
    mpl._get_executable_info('gs')
    mpl._get_executable_info('pdftops')
    if eps:
        paper_option = ['-dEPSCrop']
    elif ptype == 'figure':
        paper_option = [f'-dDEVICEWIDTHPOINTS#{bbox[2]}', f'-dDEVICEHEIGHTPOINTS#{bbox[3]}']
    else:
        paper_option = [f'-sPAPERSIZE#{ptype}']
    with TemporaryDirectory() as tmpdir:
        tmppdf = pathlib.Path(tmpdir, 'tmp.pdf')
        tmpps = pathlib.Path(tmpdir, 'tmp.ps')
        cbook._check_and_log_subprocess(['ps2pdf', '-dAutoFilterColorImages#false', '-dAutoFilterGrayImages#false', '-sAutoRotatePages#None', '-sGrayImageFilter#FlateEncode', '-sColorImageFilter#FlateEncode', *paper_option, tmpfile, tmppdf], _log)
        cbook._check_and_log_subprocess(['pdftops', '-paper', 'match', '-level3', tmppdf, tmpps], _log)
        shutil.move(tmpps, tmpfile)
    if eps:
        pstoeps(tmpfile)

@_api.deprecated('3.9')
def get_bbox_header(lbrt, rotated=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a postscript header string for the given bbox lbrt=(l, b, r, t).\n    Optionally, return rotate command.\n    '
    return (_get_bbox_header(lbrt), _get_rotate_command(lbrt) if rotated else '')

def _get_bbox_header(lbrt):
    if False:
        print('Hello World!')
    'Return a PostScript header string for bounding box *lbrt*=(l, b, r, t).'
    (l, b, r, t) = lbrt
    return f'%%BoundingBox: {int(l)} {int(b)} {math.ceil(r)} {math.ceil(t)}\n%%HiResBoundingBox: {l:.6f} {b:.6f} {r:.6f} {t:.6f}'

def _get_rotate_command(lbrt):
    if False:
        return 10
    'Return a PostScript 90° rotation command for bounding box *lbrt*=(l, b, r, t).'
    (l, b, r, t) = lbrt
    return f'{l + r:.2f} {0:.2f} translate\n90 rotate'

def pstoeps(tmpfile, bbox=None, rotated=False):
    if False:
        i = 10
        return i + 15
    '\n    Convert the postscript to encapsulated postscript.  The bbox of\n    the eps file will be replaced with the given *bbox* argument. If\n    None, original bbox will be used.\n    '
    epsfile = tmpfile + '.eps'
    with open(epsfile, 'wb') as epsh, open(tmpfile, 'rb') as tmph:
        write = epsh.write
        for line in tmph:
            if line.startswith(b'%!PS'):
                write(b'%!PS-Adobe-3.0 EPSF-3.0\n')
                if bbox:
                    write(_get_bbox_header(bbox).encode('ascii') + b'\n')
            elif line.startswith(b'%%EndComments'):
                write(line)
                write(b'%%BeginProlog\nsave\ncountdictstack\nmark\nnewpath\n/showpage {} def\n/setpagedevice {pop} def\n%%EndProlog\n%%Page 1 1\n')
                if rotated:
                    write(_get_rotate_command(bbox).encode('ascii') + b'\n')
                break
            elif bbox and line.startswith((b'%%Bound', b'%%HiResBound', b'%%DocumentMedia', b'%%Pages')):
                pass
            else:
                write(line)
        for line in tmph:
            if line.startswith(b'%%EOF'):
                write(b'cleartomark\ncountdictstack\nexch sub { end } repeat\nrestore\nshowpage\n%%EOF\n')
            elif line.startswith(b'%%PageBoundingBox'):
                pass
            else:
                write(line)
    os.remove(tmpfile)
    shutil.move(epsfile, tmpfile)
FigureManagerPS = FigureManagerBase
_psDefs = ['/_d { bind def } bind def', '/m { moveto } _d', '/l { lineto } _d', '/r { rlineto } _d', '/c { curveto } _d', '/cl { closepath } _d', '/ce { closepath eofill } _d', '/sc { setcachedevice } _d']

@_Backend.export
class _BackendPS(_Backend):
    backend_version = 'Level II'
    FigureCanvas = FigureCanvasPS