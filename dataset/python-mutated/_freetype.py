import sys
import numpy as np
from ._vispy_fonts import _vispy_fonts, _get_vispy_font_filename
if sys.platform.startswith('linux'):
    from ...ext.fontconfig import find_font
elif sys.platform.startswith('win'):
    from ._win32 import find_font
else:
    raise NotImplementedError
_font_dict = {}

def _load_font(face, bold, italic):
    if False:
        i = 10
        return i + 15
    from freetype import Face, FT_FACE_FLAG_SCALABLE
    key = '%s-%s-%s' % (face, bold, italic)
    if key in _font_dict:
        return _font_dict[key]
    if face in _vispy_fonts:
        fname = _get_vispy_font_filename(face, bold, italic)
    else:
        fname = find_font(face, bold, italic)
    font = Face(fname)
    if FT_FACE_FLAG_SCALABLE & font.face_flags == 0:
        raise RuntimeError('Font %s is not scalable, so cannot be loaded' % face)
    _font_dict[key] = font
    return font

def _load_glyph(f, char, glyphs_dict):
    if False:
        return 10
    'Load glyph from font into dict'
    from freetype import FT_LOAD_RENDER, FT_LOAD_NO_HINTING, FT_LOAD_NO_AUTOHINT
    flags = FT_LOAD_RENDER | FT_LOAD_NO_HINTING | FT_LOAD_NO_AUTOHINT
    face = _load_font(f['face'], f['bold'], f['italic'])
    face.set_char_size(f['size'] * 64)
    face.load_char(char, flags)
    bitmap = face.glyph.bitmap
    width = face.glyph.bitmap.width
    height = face.glyph.bitmap.rows
    bitmap = np.array(bitmap.buffer)
    w0 = bitmap.size // height if bitmap.size > 0 else 0
    bitmap.shape = (height, w0)
    bitmap = bitmap[:, :width].astype(np.ubyte)
    left = face.glyph.bitmap_left
    top = face.glyph.bitmap_top
    advance = face.glyph.advance.x / 64.0
    glyph = dict(char=char, offset=(left, top), bitmap=bitmap, advance=advance, kerning={})
    glyphs_dict[char] = glyph
    for (other_char, other_glyph) in glyphs_dict.items():
        kerning = face.get_kerning(other_char, char)
        glyph['kerning'][other_char] = kerning.x / 64.0
        kerning = face.get_kerning(char, other_char)
        other_glyph['kerning'][char] = kerning.x / 64.0