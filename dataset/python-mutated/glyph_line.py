"""
    In typography, a glyph /ɡlɪf/ is an elemental symbol within an agreed set of symbols,
    intended to represent a readable character for the purposes of writing.
    Glyphs are considered to be unique marks that collectively add up to the spelling of a word
    or contribute to a specific meaning of what is written, with that meaning dependent on cultural and social usage.
"""
import typing
from decimal import Decimal
from borb.io.read.types import Decimal as bDecimal
from borb.pdf.canvas.font.font import Font

class Glyph:
    """
    In typography, a glyph /ɡlɪf/ is an elemental symbol within an agreed set of symbols,
    intended to represent a readable character for the purposes of writing.
    Glyphs are considered to be unique marks that collectively add up to the spelling of a word
    or contribute to a specific meaning of what is written, with that meaning dependent on cultural and social usage.
    """

    def __init__(self, character_code: int, unicode_str: str, width: Decimal):
        if False:
            while True:
                i = 10
        self._character_code: int = character_code
        self._unicode_str: str = unicode_str
        self._width: Decimal = width

    def get_character_code(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        This function returns the character code of this Glyph object\n        '
        return self._character_code

    def get_unicode_str(self) -> str:
        if False:
            return 10
        '\n        This function returns the unicode str that this Glyph represents\n        '
        return self._unicode_str

    def get_width(self) -> Decimal:
        if False:
            while True:
                i = 10
        '\n        This function returns the width (in text space) of this Glyph\n        '
        return self._width

class GlyphLine:
    """
    This class represents a line of Glyph objects.
    This class contains utility methods to work with collections of Glyph objects.
    """

    def __init__(self, glyphs: typing.List[Glyph], font: Font, font_size: Decimal, character_spacing: Decimal=Decimal(0), word_spacing: Decimal=Decimal(0), horizontal_scaling: Decimal=Decimal(100)):
        if False:
            i = 10
            return i + 15
        assert isinstance(font, Font)
        self._glyphs: typing.List[Glyph] = glyphs
        self._font = font
        self._font_size = font_size
        self._character_spacing = character_spacing
        self._word_spacing = word_spacing
        self._horizontal_scaling = horizontal_scaling

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._glyphs)

    def __str__(self):
        if False:
            print('Hello World!')
        return self.get_text()

    @staticmethod
    def _isspace(c: str) -> bool:
        if False:
            while True:
                i = 10
        return ord(c) in [9, 10, 11, 12, 13, 32]

    def add(self, glyph_or_glyphline: typing.Union[Glyph, 'GlyphLine']) -> 'GlyphLine':
        if False:
            while True:
                i = 10
        '\n        This function appends a Glyph (or all Glyph objects in a GlyphLine) to this GlyphLine.\n        This function returns self.\n        '
        if isinstance(glyph_or_glyphline, Glyph):
            self._glyphs.append(glyph_or_glyphline)
        if isinstance(glyph_or_glyphline, GlyphLine):
            for g in glyph_or_glyphline._glyphs:
                self._glyphs.append(g)
        return self

    @staticmethod
    def from_bytes(text: bytes, font: Font, font_size: Decimal, character_spacing: Decimal=Decimal(0), word_spacing: Decimal=Decimal(0), horizontal_scaling: Decimal=Decimal(100)) -> 'GlyphLine':
        if False:
            for i in range(10):
                print('nop')
        '\n        This method constructs a new GlyphLine from text (represented as character ids)\n        :param text:                a byte-array containing the character ids\n        :param font:                the font\n        :param font_size:           the font-size\n        :param character_spacing:   the (additional) space between characters\n        :param word_spacing:        the (additional) space between words\n        :param horizontal_scaling:  the horizontal scaling factor (100 represents no zoom)\n        :return:                    a GlyphLine\n        '
        glyphs: typing.List[Glyph] = []
        i: int = 0
        while i < len(text):
            unicode_chars: typing.Optional[str] = None
            if i + 1 < len(text):
                multi_byte_char_code: int = text[i] * 256 + text[i + 1]
                unicode_chars = font.character_identifier_to_unicode(multi_byte_char_code)
                if unicode_chars is not None:
                    glyphs.append(Glyph(multi_byte_char_code, unicode_chars, font.get_width(multi_byte_char_code) or bDecimal(0)))
                    i += 2
                    continue
            if i < len(text):
                unicode_chars = font.character_identifier_to_unicode(text[i])
                if unicode_chars is not None:
                    glyphs.append(Glyph(text[i], unicode_chars, font.get_width(text[i]) or Decimal(0)))
                    i += 1
                    continue
            if i < len(text):
                glyphs.append(Glyph(text[i], '�', Decimal(250)))
                i += 1
        return GlyphLine(glyphs, font, font_size, character_spacing, word_spacing, horizontal_scaling)

    @staticmethod
    def from_str(text: str, font: Font, font_size: Decimal, character_spacing: Decimal=Decimal(0), word_spacing: Decimal=Decimal(0), horizontal_scaling: Decimal=Decimal(100)) -> 'GlyphLine':
        if False:
            print('Hello World!')
        '\n        This method constructs a new GlyphLine from text\n        :param text:                a string which will be decoded into character-ids\n        :param font:                the font\n        :param font_size:           the font-size\n        :param character_spacing:   the (additional) space between characters\n        :param word_spacing:        the (additional) space between words\n        :param horizontal_scaling:  the horizontal scaling factor (100 represents no zoom)\n        :return:                    a GlyphLine\n        '
        character_ids: typing.List[int] = [font.unicode_to_character_identifier(c) or 0 for c in text]
        glyphs: typing.List[Glyph] = [Glyph(cid, text[i], font.get_width(cid) or Decimal(0)) for (i, cid) in enumerate(character_ids)]
        return GlyphLine(glyphs, font, font_size, character_spacing, word_spacing, horizontal_scaling)

    def get_text(self) -> str:
        if False:
            print('Hello World!')
        '\n        This function returns the unicode str represented by the Glyph objects in this GlyphLine\n        '
        return ''.join([x.get_unicode_str() for x in self._glyphs])

    def get_width_in_text_space(self) -> Decimal:
        if False:
            return 10
        '\n        This function calculates the width (in text space) of this GlyphLine\n        '
        w: Decimal = Decimal(0)
        for g in self._glyphs:
            glyph_width_in_text_space = g.get_width() * self._font_size * Decimal(0.001)
            if len(g.get_unicode_str()) == 1 and GlyphLine._isspace(g.get_unicode_str()):
                glyph_width_in_text_space += self._word_spacing
            glyph_width_in_text_space *= self._horizontal_scaling / Decimal(100)
            glyph_width_in_text_space += self._character_spacing
            w += glyph_width_in_text_space
        w -= self._character_spacing
        return w

    def split(self) -> typing.List['GlyphLine']:
        if False:
            return 10
        '\n        This function splits the GlyphLine into several GlyphLine objects,\n        one per Glyph in the (original, this) GlyphLine.\n        '
        out: typing.List['GlyphLine'] = []
        for g in self._glyphs:
            out.append(GlyphLine(b'', self._font, self._font_size, self._character_spacing, self._word_spacing, self._horizontal_scaling))
            out[-1]._glyphs = [g]
        return out

    def uses_descent(self) -> bool:
        if False:
            print('Hello World!')
        '\n        This function returns True if any of the Glyph objects in the GlyphLine has a non-zero descent, False otherwise\n        '
        return any([x in ['y', 'p', 'q', 'f', 'g', 'j'] for x in self.get_text()])