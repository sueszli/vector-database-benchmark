"""
Low-level text helper utilities.
"""
import dataclasses
from . import _api
from .ft2font import KERNING_DEFAULT, LOAD_NO_HINTING
LayoutItem = dataclasses.make_dataclass('LayoutItem', ['ft_object', 'char', 'glyph_idx', 'x', 'prev_kern'])

def warn_on_missing_glyph(codepoint, fontnames):
    if False:
        for i in range(10):
            print('nop')
    _api.warn_external(f"Glyph {codepoint} ({chr(codepoint).encode('ascii', 'namereplace').decode('ascii')}) missing from font(s) {fontnames}.")
    block = 'Hebrew' if 1424 <= codepoint <= 1535 else 'Arabic' if 1536 <= codepoint <= 1791 else 'Devanagari' if 2304 <= codepoint <= 2431 else 'Bengali' if 2432 <= codepoint <= 2559 else 'Gurmukhi' if 2560 <= codepoint <= 2687 else 'Gujarati' if 2688 <= codepoint <= 2815 else 'Oriya' if 2816 <= codepoint <= 2943 else 'Tamil' if 2944 <= codepoint <= 3071 else 'Telugu' if 3072 <= codepoint <= 3199 else 'Kannada' if 3200 <= codepoint <= 3327 else 'Malayalam' if 3328 <= codepoint <= 3455 else 'Sinhala' if 3456 <= codepoint <= 3583 else None
    if block:
        _api.warn_external(f'Matplotlib currently does not support {block} natively.')

def layout(string, font, *, kern_mode=KERNING_DEFAULT):
    if False:
        return 10
    "\n    Render *string* with *font*.  For each character in *string*, yield a\n    (glyph-index, x-position) pair.  When such a pair is yielded, the font's\n    glyph is set to the corresponding character.\n\n    Parameters\n    ----------\n    string : str\n        The string to be rendered.\n    font : FT2Font\n        The font.\n    kern_mode : int\n        A FreeType kerning mode.\n\n    Yields\n    ------\n    glyph_index : int\n    x_position : float\n    "
    x = 0
    prev_glyph_idx = None
    char_to_font = font._get_fontmap(string)
    base_font = font
    for char in string:
        font = char_to_font.get(char, base_font)
        glyph_idx = font.get_char_index(ord(char))
        kern = base_font.get_kerning(prev_glyph_idx, glyph_idx, kern_mode) / 64 if prev_glyph_idx is not None else 0.0
        x += kern
        glyph = font.load_glyph(glyph_idx, flags=LOAD_NO_HINTING)
        yield LayoutItem(font, char, glyph_idx, x, kern)
        x += glyph.linearHoriAdvance / 65536
        prev_glyph_idx = glyph_idx