"""
Type 3 fonts differ from the other fonts supported by PDF. A Type 3 font dictionary defines the font; font
dictionaries for other fonts simply contain information about the font and refer to a separate font program for the
actual glyph descriptions. In Type 3 fonts, glyphs shall be defined by streams of PDF graphics operators. These
streams shall be associated with glyph names. A separate encoding entry shall map character codes to the
appropriate glyph names for the glyphs.
"""
import logging
import typing
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Name
from borb.pdf.canvas.font.font import Font
from borb.pdf.canvas.font.simple_font.font_type_1 import Type1Font
logger = logging.getLogger(__name__)

class Type3Font(Type1Font):
    """
    Type 3 fonts differ from the other fonts supported by PDF. A Type 3 font dictionary defines the font; font
    dictionaries for other fonts simply contain information about the font and refer to a separate font program for the
    actual glyph descriptions. In Type 3 fonts, glyphs shall be defined by streams of PDF graphics operators. These
    streams shall be associated with glyph names. A separate encoding entry shall map character codes to the
    appropriate glyph names for the glyphs.
    """

    def __init__(self):
        if False:
            return 10
        super(Type3Font, self).__init__()
        self[Name('Subtype')] = Name('Type3')

    def __deepcopy__(self, memodict={}):
        if False:
            while True:
                i = 10
        f_out: Font = super(Type3Font, self).__deepcopy__(memodict)
        f_out[Name('Subtype')] = Name('Type3')
        f_out._character_identifier_to_unicode_lookup: typing.Dict[int, str] = {k: v for (k, v) in self._character_identifier_to_unicode_lookup.items()}
        f_out._unicode_lookup_to_character_identifier: typing.Dict[str, int] = {k: v for (k, v) in self._unicode_lookup_to_character_identifier.items()}
        return f_out

    def _empty_copy(self) -> 'Font':
        if False:
            i = 10
            return i + 15
        return Type3Font()

    def get_ascent(self) -> bDecimal:
        if False:
            while True:
                i = 10
        '\n        This function returns the maximum height above the baseline reached by glyphs in this font.\n        The height of glyphs for accented characters shall be excluded.\n        '
        if 'FontDescriptor' in self and 'Ascent' in 'FontDescriptor':
            return self['FontDescriptor']['Ascent']
        logger.debug('Type3Font does not have an `Ascent` entry in its `FontDescriptor` dictionary.')
        return bDecimal(0)

    def get_descent(self) -> bDecimal:
        if False:
            i = 10
            return i + 15
        '\n        This function returns the maximum depth below the baseline reached by glyphs in this font.\n        The value shall be a negative number.\n        '
        if 'FontDescriptor' in self and 'Descent' in 'FontDescriptor':
            return self['FontDescriptor']['Descent']
        logger.debug('Type3Font does not have an `Descent` entry in its `FontDescriptor` dictionary.')
        return bDecimal(0)