"""
    CID-keyed fonts provide a convenient and efficient method for defining multiple-byte character encodings and
   fonts with a large number of glyphs. These capabilities provide great flexibility for representing text in writing
    systems for languages with large character sets, such as Chinese, Japanese, and Korean (CJK).

    The CID-keyed font architecture specifies the external representation of certain font programs, called CMap
    and CIDFont files, along with some conventions for combining and using those files. As mentioned earlier, PDF
    does not support the entire CID-keyed font architecture, which is independent of PDF; CID-keyed fonts may be
    used in other environments.

    A Type 2 CIDFont contains glyph descriptions based on the TrueType font format
"""
import typing
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Name
from borb.pdf.canvas.font.composite_font.cid_font_type_0 import CIDType0Font
from borb.pdf.canvas.font.font import Font

class CIDType2Font(CIDType0Font):
    """
    A Type 2 CIDFont contains glyph descriptions based on the TrueType font format
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(CIDType2Font, self).__init__()
        self._cid_to_gid_map_cache: typing.Dict[int, int] = {}

    def __deepcopy__(self, memodict={}):
        if False:
            return 10
        f_out: CIDType2Font = super(CIDType2Font, self).__deepcopy__(memodict)
        f_out[Name('Subtype')] = Name('CIDFontType2')
        f_out._width_cache: typing.Dict[int, bDecimal] = {k: v for (k, v) in self._width_cache.items()}
        return f_out

    def _empty_copy(self) -> 'Font':
        if False:
            for i in range(10):
                print('nop')
        return CIDType2Font()