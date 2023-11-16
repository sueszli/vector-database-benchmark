"""
A Type 1 font program is a stylized PostScript program that describes glyph shapes. It uses a compact
encoding for the glyph descriptions, and it includes hint information that enables high-quality rendering even at
small sizes and low resolutions.
"""
import copy
import io
import logging
import typing
from pathlib import Path
from fontTools.afmLib import AFM
from fontTools.agl import toUnicode
from fontTools.cffLib import CFFFontSet
from fontTools.cffLib import TopDict
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Dictionary
from borb.io.read.types import Name
from borb.pdf.canvas.font.adobe_glyph_list import AdobeGlyphList
from borb.pdf.canvas.font.adobe_standard_encoding import adobe_standard_decode
from borb.pdf.canvas.font.adobe_standard_encoding import adobe_standard_encode
from borb.pdf.canvas.font.font import Font
from borb.pdf.canvas.font.simple_font.simple_font import SimpleFont
from borb.pdf.canvas.font.symbol_encoding import symbol_decode
from borb.pdf.canvas.font.symbol_encoding import zapfdingbats_decode
logger = logging.getLogger(__name__)

class Type1Font(SimpleFont):
    """
    A Type 1 font program is a stylized PostScript program that describes glyph shapes. It uses a compact
    encoding for the glyph descriptions, and it includes hint information that enables high-quality rendering even at
    small sizes and low resolutions.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(Type1Font, self).__init__()
        self[Name('Type')] = Name('Font')
        self[Name('Subtype')] = Name('Type1')
        self._character_identifier_to_unicode_lookup: typing.Dict[int, str] = {}
        self._unicode_lookup_to_character_identifier: typing.Dict[str, int] = {}

    def __deepcopy__(self, memodict={}):
        if False:
            return 10
        f_out: Font = super(Type1Font, self).__deepcopy__(memodict)
        f_out[Name('Subtype')] = Name('Type1')
        f_out._character_identifier_to_unicode_lookup: typing.Dict[int, str] = {k: v for (k, v) in self._character_identifier_to_unicode_lookup.items()}
        f_out._unicode_lookup_to_character_identifier: typing.Dict[str, int] = {k: v for (k, v) in self._unicode_lookup_to_character_identifier.items()}
        return f_out

    def _empty_copy(self) -> 'Font':
        if False:
            while True:
                i = 10
        return Type1Font()

    def _read_encoding_with_differences(self) -> None:
        if False:
            while True:
                i = 10
        if len(self._unicode_lookup_to_character_identifier) > 0:
            return
        if all([str(x).startswith('G') and toUnicode(str(x)) == '' for x in self['Encoding']['Differences'] if isinstance(x, Name)]):
            self._read_encoding_with_unclear_glyph_names()
            return
        assert 'FirstChar' in self, 'Type1Font must have a /FirstChar entry'
        assert isinstance(self['FirstChar'], bDecimal), 'Type1Font must have a valid /FirstChar entry'
        assert 'LastChar' in self, 'Type1Font must have a /LastChar entry'
        assert isinstance(self['LastChar'], bDecimal), 'Type1Font must have a valid /LastChar entry'
        first_char: int = int(self['FirstChar'])
        last_char: int = int(self['LastChar'])
        self._character_identifier_to_unicode_lookup = {}
        i: int = 0
        for i in range(first_char, last_char + 1):
            y: typing.Optional[str] = None
            try:
                if self['Encoding']['BaseEncoding'] == 'WinAnsiEncoding':
                    y = bytes([i]).decode('cp1252')
                elif self['Encoding']['BaseEncoding'] == 'MacRomanEncoding':
                    y = bytes([i]).decode('mac-roman')
                elif self['Encoding']['BaseEncoding'] == 'MacExpertEncoding':
                    logger.debug('Font %s uses MacExpertEncoding, defaulting to MacRomanEncoding' % str(self['BaseFont']))
                    y = bytes([i]).decode('mac-roman')
                elif self['Encoding']['BaseEncoding'] == 'StandardEncoding':
                    y = adobe_standard_decode(bytes([i]))
            except:
                pass
            if y is not None:
                self._character_identifier_to_unicode_lookup[i] = y
        if 'FontDescriptor' in self and 'FontFile3' in self['FontDescriptor']:
            font_file_bytes: bytes = self['FontDescriptor']['FontFile3']['DecodedBytes']
            cff: CFFFontSet = CFFFontSet()
            cff.major = 1
            cff.decompile(io.BytesIO(font_file_bytes), otFont=None)
            assert len(cff.keys()) == 1
            top_level_dict: TopDict = cff[0]
        j: int = 0
        i = 0
        while i < len(self['Encoding']['Differences']):
            assert isinstance(self['Encoding']['Differences'][i], bDecimal)
            character_code: int = self['Encoding']['Differences'][i]
            j = i + 1
            while j < len(self['Encoding']['Differences']) and (not isinstance(self['Encoding']['Differences'][j], bDecimal)):
                glyph_name: str = str(self['Encoding']['Differences'][j])
                self._character_identifier_to_unicode_lookup[int(character_code)] = toUnicode(glyph_name)
                character_code += 1
                j += 1
            i = j
        self._unicode_lookup_to_character_identifier = {v: k for (k, v) in self._character_identifier_to_unicode_lookup.items()}

    def _read_encoding_with_unclear_glyph_names(self):
        if False:
            print('Hello World!')
        assert 'FirstChar' in self, 'Type1Font must have a /FirstChar entry'
        assert isinstance(self['FirstChar'], bDecimal), 'Type1Font must have a valid /FirstChar entry'
        assert 'LastChar' in self, 'Type1Font must have a /LastChar entry'
        assert isinstance(self['LastChar'], bDecimal), 'Type1Font must have a valid /LastChar entry'
        first_char: int = int(self['FirstChar'])
        last_char: int = int(self['LastChar'])
        self._character_identifier_to_unicode_lookup = {}
        i: int = 0
        for i in range(first_char, last_char + 1):
            y: typing.Optional[str] = None
            try:
                if self['Encoding']['BaseEncoding'] == 'WinAnsiEncoding':
                    y = bytes([i]).decode('cp1252')
                elif self['Encoding']['BaseEncoding'] == 'MacRomanEncoding':
                    y = bytes([i]).decode('mac-roman')
                elif self['Encoding']['BaseEncoding'] == 'MacExpertEncoding':
                    logger.debug('Font %s uses MacExpertEncoding, defaulting to MacRomanEncoding' % str(self['BaseFont']))
                    y = bytes([i]).decode('mac-roman')
                elif self['Encoding']['BaseEncoding'] == 'StandardEncoding':
                    y = adobe_standard_decode(bytes([i]))
            except:
                pass
            if y is not None:
                self._character_identifier_to_unicode_lookup[i] = y
        j: int = 0
        i = 0
        encoding_without_differences: typing.Dict[int, str] = copy.deepcopy(self._character_identifier_to_unicode_lookup)
        while i < len(self['Encoding']['Differences']):
            assert isinstance(self['Encoding']['Differences'][i], bDecimal)
            character_code: int = self['Encoding']['Differences'][i]
            j = i + 1
            while j < len(self['Encoding']['Differences']) and (not isinstance(self['Encoding']['Differences'][j], bDecimal)):
                glyph_name: str = str(self['Encoding']['Differences'][j])
                cid: int = int(glyph_name[1:])
                self._character_identifier_to_unicode_lookup[int(character_code)] = encoding_without_differences[cid]
                character_code += 1
                j += 1
            i = j
        self._unicode_lookup_to_character_identifier = {v: k for (k, v) in self._character_identifier_to_unicode_lookup.items()}

    def _read_to_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self._unicode_lookup_to_character_identifier) > 0:
            return
        assert 'ToUnicode' in self, 'Type1Font must have a /ToUnicode entry.'
        assert 'DecodedBytes' in self['ToUnicode'], 'Type1Font must have a valid /ToUnicode entry.'
        cmap_bytes: bytes = self['ToUnicode']['DecodedBytes']
        self._character_identifier_to_unicode_lookup = self._read_cmap(cmap_bytes)
        self._unicode_lookup_to_character_identifier: typing.Dict[str, int] = {v: k for (k, v) in self._character_identifier_to_unicode_lookup.items()}

    def character_identifier_to_unicode(self, character_identifier: int) -> typing.Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        This function maps a character identifier to its unicode str.\n        If no such mapping exists, this function returns None.\n        '
        if Name('ToUnicode') in self:
            self._read_to_unicode()
            return self._character_identifier_to_unicode_lookup.get(character_identifier)
        if 'Encoding' not in self:
            self[Name('Encoding')] = Name('StandardEncoding')
        if 'Encoding' in self and isinstance(self['Encoding'], Dictionary) and ('BaseEncoding' not in self['Encoding']):
            self['Encoding'][Name('BaseEncoding')] = Name('WinAnsiEncoding')
        if isinstance(self['Encoding'], Name) and self['Encoding'] in ['MacRomanEncoding', 'MacExpertEncoding', 'WinAnsiEncoding', 'StandardEncoding']:
            if character_identifier < 0 or character_identifier > 256:
                return None
            try:
                if self['Encoding'] == 'WinAnsiEncoding':
                    return bytes([character_identifier]).decode('cp1252')
                elif self['Encoding'] == 'MacRomanEncoding':
                    return bytes([character_identifier]).decode('mac-roman')
                elif self['Encoding'] == 'MacExpertEncoding':
                    logger.debug('Font %s uses MacExpertEncoding, defaulting to MacRomanEncoding' % str(self['BaseFont']))
                    return bytes([character_identifier]).decode('mac-roman')
                elif self['Encoding'] == 'StandardEncoding':
                    return adobe_standard_decode(bytes([character_identifier]))
                else:
                    logger.debug('Font %s uses unknown encoding %s' % (str(self['BaseFont']), str(self['Encoding'])))
            except UnicodeDecodeError:
                return None
        if isinstance(self['Encoding'], Dictionary) and 'BaseEncoding' in self['Encoding'] and (self['Encoding']['BaseEncoding'] in ['MacRomanEncoding', 'MacExpertEncoding', 'WinAnsiEncoding', 'StandardEncoding']):
            self._read_encoding_with_differences()
            if character_identifier < 0 or character_identifier > 256:
                return None
            return self._character_identifier_to_unicode_lookup.get(character_identifier)
        return None

    def get_ascent(self) -> bDecimal:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns the maximum height above the baseline reached by glyphs in this font.\n        The height of glyphs for accented characters shall be excluded.\n        '
        return self['FontDescriptor']['Ascent']

    def get_descent(self) -> bDecimal:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns the maximum depth below the baseline reached by glyphs in this font.\n        The value shall be a negative number.\n        '
        return self['FontDescriptor']['Descent']

    def get_width(self, character_identifier: int) -> typing.Optional[bDecimal]:
        if False:
            i = 10
            return i + 15
        '\n        This function returns the width (in text space) of a given character identifier.\n        If this Font is unable to represent the glyph that corresponds to the character identifier,\n        this function returns None\n        '
        first_char: int = int(self['FirstChar'])
        last_char: int = int(self['LastChar'])
        if first_char <= character_identifier <= last_char:
            return self['Widths'][character_identifier - first_char]
        return None

    def unicode_to_character_identifier(self, unicode: str) -> typing.Optional[int]:
        if False:
            i = 10
            return i + 15
        '\n        This function maps a unicode str to its character identifier.\n        If no such mapping exists, this function returns None.\n        '
        if Name('ToUnicode') in self:
            self._read_to_unicode()
            return self._unicode_lookup_to_character_identifier.get(unicode)
        if 'Encoding' not in self:
            self[Name('Encoding')] = Name('StandardEncoding')
        if isinstance(self['Encoding'], Name) and self['Encoding'] in ['MacRomanEncoding', 'MacExpertEncoding', 'WinAnsiEncoding', 'StandardEncoding']:
            try:
                if self['Encoding'] == 'WinAnsiEncoding':
                    return int(unicode.encode('cp1252')[0])
                elif self['Encoding'] == 'MacRomanEncoding':
                    return int(unicode.encode('mac-roman')[0])
                elif self['Encoding'] == 'MacExpertEncoding':
                    return int(unicode.encode('mac-roman')[0])
                elif self['Encoding'] == 'StandardEncoding':
                    return int(adobe_standard_encode(unicode)[0])
            except:
                return None
        if isinstance(self['Encoding'], Dictionary) and 'BaseEncoding' in self['Encoding'] and (self['Encoding']['BaseEncoding'] in ['MacRomanEncoding', 'MacExpertEncoding', 'WinAnsiEncoding', 'StandardEncoding']):
            self._read_encoding_with_differences()
            return self._unicode_lookup_to_character_identifier.get(unicode, None)
        return None

class StandardType1Font(Type1Font):
    """
    The PostScript names of 14 Type 1 fonts, known as the standard 14 fonts, are as follows: Times-Roman,
    Helvetica, Courier, Symbol, Times-Bold, Helvetica-Bold, Courier-Bold, ZapfDingbats, Times-Italic, Helvetica-
    Oblique, Courier-Oblique, Times-BoldItalic, Helvetica-BoldOblique, Courier-BoldOblique
    These fonts, or their font metrics and suitable substitution fonts, shall be available to the conforming reader.
    """
    STANDARD_14_FONT_NAMES: typing.List[str] = ['Courier', 'Courier-Bold', 'Courier-Bold-Oblique', 'Courier-Oblique', 'Helvetica', 'Helvetica-Bold', 'Helvetica-Bold-Oblique', 'Helvetica-Oblique', 'Symbol', 'Times-Bold', 'Times-Bold-Italic', 'Times-Italic', 'Times-Roman', 'ZapfDingbats']

    def __init__(self, font_name: typing.Optional[str]=None):
        if False:
            i = 10
            return i + 15
        super(StandardType1Font, self).__init__()
        if font_name is not None:
            font_name = StandardType1Font._canonical_name(font_name)
            assert font_name is not None, 'font_name must be one of the 14 StandardType1Font names.'
            afm_directory: Path = Path(__file__).parent / 'afm'
            assert afm_directory.exists(), 'AFM directory not found'
            afm_file: Path = afm_directory / (font_name.lower() + '.afm')
            assert afm_file.exists(), 'afm file not found'
            self._afm: AFM = AFM(afm_file)
            self[Name('Type')] = Name('Font')
            self[Name('Subtype')] = Name('Type1')
            self[Name('BaseFont')] = Name(self._afm._attrs['FontName'])
            self._character_identifier_to_unicode_lookup: typing.Dict[int, str] = {}
            self._unicode_lookup_to_character_identifier: typing.Dict[str, int] = {}
            if font_name == 'Symbol':
                self._character_identifier_to_unicode_lookup = {c: symbol_decode(bytes([c])) for c in range(0, 256)}
                self._unicode_lookup_to_character_identifier = {v: k for (k, v) in self._character_identifier_to_unicode_lookup.items()}
            elif font_name == 'ZapfDingbats':
                self._character_identifier_to_unicode_lookup = {c: zapfdingbats_decode(bytes([c])) for c in range(0, 256)}
                self._unicode_lookup_to_character_identifier = {v: k for (k, v) in self._character_identifier_to_unicode_lookup.items()}
            else:
                self[Name('Encoding')] = Name('WinAnsiEncoding')
                for c in range(0, 256):
                    try:
                        self._character_identifier_to_unicode_lookup[c] = bytes([c]).decode('cp1252')
                    except:
                        self._character_identifier_to_unicode_lookup[c] = ''
                self._unicode_lookup_to_character_identifier = {v: k for (k, v) in self._character_identifier_to_unicode_lookup.items()}

    def __deepcopy__(self, memodict={}):
        if False:
            print('Hello World!')
        f_out: Font = super(StandardType1Font, self).__deepcopy__(memodict)
        f_out[Name('Subtype')] = Name('Type1')
        f_out._character_identifier_to_unicode_lookup: typing.Dict[int, str] = {k: v for (k, v) in self._character_identifier_to_unicode_lookup.items()}
        f_out._unicode_lookup_to_character_identifier: typing.Dict[str, int] = {k: v for (k, v) in self._unicode_lookup_to_character_identifier.items()}
        f_out._afm = self._afm
        return f_out

    @staticmethod
    def _canonical_name(font_name: str) -> typing.Optional[str]:
        if False:
            return 10

        def _to_lower_and_alpha(x: str) -> str:
            if False:
                i = 10
                return i + 15
            return ''.join([c for c in x.lower() if c in 'abcdefghijklmnopqrstuvwxyz'])
        canonical_name: str = _to_lower_and_alpha(font_name)
        for n in StandardType1Font.STANDARD_14_FONT_NAMES:
            if _to_lower_and_alpha(n) == canonical_name:
                return n
        return None

    def _empty_copy(self) -> 'Font':
        if False:
            return 10
        return StandardType1Font()

    def character_identifier_to_unicode(self, character_identifier: int) -> typing.Optional[str]:
        if False:
            while True:
                i = 10
        '\n        This function maps a character identifier to its unicode str.\n        If no such mapping exists, this function returns None.\n        '
        return self._character_identifier_to_unicode_lookup.get(character_identifier)

    def get_ascent(self) -> bDecimal:
        if False:
            return 10
        '\n        This function returns the maximum height above the baseline reached by glyphs in this font.\n        The height of glyphs for accented characters shall be excluded.\n        '
        if 'Ascender' in self._afm._attrs:
            return bDecimal(self._afm._attrs['Ascender'])
        return bDecimal(0)

    def get_descent(self) -> bDecimal:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns the maximum depth below the baseline reached by glyphs in this font.\n        The value shall be a negative number.\n        '
        if 'Descender' in self._afm._attrs:
            return bDecimal(self._afm._attrs['Descender'])
        return bDecimal(0)

    def get_width(self, character_identifier: int) -> typing.Optional[bDecimal]:
        if False:
            print('Hello World!')
        '\n        This function returns the width (in text space) of a given character identifier.\n        If this Font is unable to represent the glyph that corresponds to the character identifier,\n        this function returns None\n        '
        default_tuple: typing.Tuple[int, int, typing.Tuple[int, int, int, int]] = (0, 0, (0, 0, 0, 0))
        name: typing.Optional[str] = AdobeGlyphList.UNICODE_TO_NAME.get(ord(self._character_identifier_to_unicode_lookup[character_identifier]), None)
        if name in self._afm._chars:
            return bDecimal(self._afm._chars.get(name)[1])
        if f'a{character_identifier}' in self._afm._chars:
            return bDecimal(self._afm._chars.get(f'a{character_identifier}')[1])
        return default_tuple[1]

    @staticmethod
    def is_standard_14_font_name(font_name: str) -> bool:
        if False:
            return 10
        '\n        This function returns True if the given str represents the name of one of the standard 14 fonts, False otherwise\n        '
        return StandardType1Font._canonical_name(font_name) is not None

    def unicode_to_character_identifier(self, unicode: str) -> typing.Optional[int]:
        if False:
            return 10
        '\n        This function maps a unicode str to its character identifier.\n        If no such mapping exists, this function returns None.\n        '
        return self._unicode_lookup_to_character_identifier.get(unicode)