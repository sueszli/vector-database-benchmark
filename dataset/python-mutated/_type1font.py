"""
A class representing a Type 1 font.

This version reads pfa and pfb files and splits them for embedding in
pdf files. It also supports SlantFont and ExtendFont transformations,
similarly to pdfTeX and friends. There is no support yet for subsetting.

Usage::

    font = Type1Font(filename)
    clear_part, encrypted_part, finale = font.parts
    slanted_font = font.transform({'slant': 0.167})
    extended_font = font.transform({'extend': 1.2})

Sources:

* Adobe Technical Note #5040, Supporting Downloadable PostScript
  Language Fonts.

* Adobe Type 1 Font Format, Adobe Systems Incorporated, third printing,
  v1.1, 1993. ISBN 0-201-57044-0.
"""
import binascii
import functools
import logging
import re
import string
import struct
import numpy as np
from matplotlib.cbook import _format_approx
from . import _api
_log = logging.getLogger(__name__)

class _Token:
    """
    A token in a PostScript stream.

    Attributes
    ----------
    pos : int
        Position, i.e. offset from the beginning of the data.
    raw : str
        Raw text of the token.
    kind : str
        Description of the token (for debugging or testing).
    """
    __slots__ = ('pos', 'raw')
    kind = '?'

    def __init__(self, pos, raw):
        if False:
            print('Hello World!')
        _log.debug('type1font._Token %s at %d: %r', self.kind, pos, raw)
        self.pos = pos
        self.raw = raw

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'<{self.kind} {self.raw} @{self.pos}>'

    def endpos(self):
        if False:
            while True:
                i = 10
        'Position one past the end of the token'
        return self.pos + len(self.raw)

    def is_keyword(self, *names):
        if False:
            for i in range(10):
                print('nop')
        'Is this a name token with one of the names?'
        return False

    def is_slash_name(self):
        if False:
            while True:
                i = 10
        'Is this a name token that starts with a slash?'
        return False

    def is_delim(self):
        if False:
            print('Hello World!')
        'Is this a delimiter token?'
        return False

    def is_number(self):
        if False:
            return 10
        'Is this a number token?'
        return False

    def value(self):
        if False:
            for i in range(10):
                print('nop')
        return self.raw

class _NameToken(_Token):
    kind = 'name'

    def is_slash_name(self):
        if False:
            for i in range(10):
                print('nop')
        return self.raw.startswith('/')

    def value(self):
        if False:
            while True:
                i = 10
        return self.raw[1:]

class _BooleanToken(_Token):
    kind = 'boolean'

    def value(self):
        if False:
            i = 10
            return i + 15
        return self.raw == 'true'

class _KeywordToken(_Token):
    kind = 'keyword'

    def is_keyword(self, *names):
        if False:
            return 10
        return self.raw in names

class _DelimiterToken(_Token):
    kind = 'delimiter'

    def is_delim(self):
        if False:
            print('Hello World!')
        return True

    def opposite(self):
        if False:
            return 10
        return {'[': ']', ']': '[', '{': '}', '}': '{', '<<': '>>', '>>': '<<'}[self.raw]

class _WhitespaceToken(_Token):
    kind = 'whitespace'

class _StringToken(_Token):
    kind = 'string'
    _escapes_re = re.compile('\\\\([\\\\()nrtbf]|[0-7]{1,3})')
    _replacements = {'\\': '\\', '(': '(', ')': ')', 'n': '\n', 'r': '\r', 't': '\t', 'b': '\x08', 'f': '\x0c'}
    _ws_re = re.compile('[\x00\t\r\x0c\n ]')

    @classmethod
    def _escape(cls, match):
        if False:
            print('Hello World!')
        group = match.group(1)
        try:
            return cls._replacements[group]
        except KeyError:
            return chr(int(group, 8))

    @functools.lru_cache
    def value(self):
        if False:
            return 10
        if self.raw[0] == '(':
            return self._escapes_re.sub(self._escape, self.raw[1:-1])
        else:
            data = self._ws_re.sub('', self.raw[1:-1])
            if len(data) % 2 == 1:
                data += '0'
            return binascii.unhexlify(data)

class _BinaryToken(_Token):
    kind = 'binary'

    def value(self):
        if False:
            for i in range(10):
                print('nop')
        return self.raw[1:]

class _NumberToken(_Token):
    kind = 'number'

    def is_number(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def value(self):
        if False:
            return 10
        if '.' not in self.raw:
            return int(self.raw)
        else:
            return float(self.raw)

def _tokenize(data: bytes, skip_ws: bool):
    if False:
        return 10
    '\n    A generator that produces _Token instances from Type-1 font code.\n\n    The consumer of the generator may send an integer to the tokenizer to\n    indicate that the next token should be _BinaryToken of the given length.\n\n    Parameters\n    ----------\n    data : bytes\n        The data of the font to tokenize.\n\n    skip_ws : bool\n        If true, the generator will drop any _WhitespaceTokens from the output.\n    '
    text = data.decode('ascii', 'replace')
    whitespace_or_comment_re = re.compile('[\\0\\t\\r\\f\\n ]+|%[^\\r\\n]*')
    token_re = re.compile('/{0,2}[^]\\0\\t\\r\\f\\n ()<>{}/%[]+')
    instring_re = re.compile('[()\\\\]')
    hex_re = re.compile('^<[0-9a-fA-F\\0\\t\\r\\f\\n ]*>$')
    oct_re = re.compile('[0-7]{1,3}')
    pos = 0
    next_binary = None
    while pos < len(text):
        if next_binary is not None:
            n = next_binary
            next_binary = (yield _BinaryToken(pos, data[pos:pos + n]))
            pos += n
            continue
        match = whitespace_or_comment_re.match(text, pos)
        if match:
            if not skip_ws:
                next_binary = (yield _WhitespaceToken(pos, match.group()))
            pos = match.end()
        elif text[pos] == '(':
            start = pos
            pos += 1
            depth = 1
            while depth:
                match = instring_re.search(text, pos)
                if match is None:
                    raise ValueError(f'Unterminated string starting at {start}')
                pos = match.end()
                if match.group() == '(':
                    depth += 1
                elif match.group() == ')':
                    depth -= 1
                else:
                    char = text[pos]
                    if char in '\\()nrtbf':
                        pos += 1
                    else:
                        octal = oct_re.match(text, pos)
                        if octal:
                            pos = octal.end()
                        else:
                            pass
            next_binary = (yield _StringToken(start, text[start:pos]))
        elif text[pos:pos + 2] in ('<<', '>>'):
            next_binary = (yield _DelimiterToken(pos, text[pos:pos + 2]))
            pos += 2
        elif text[pos] == '<':
            start = pos
            try:
                pos = text.index('>', pos) + 1
            except ValueError as e:
                raise ValueError(f'Unterminated hex string starting at {start}') from e
            if not hex_re.match(text[start:pos]):
                raise ValueError(f'Malformed hex string starting at {start}')
            next_binary = (yield _StringToken(pos, text[start:pos]))
        else:
            match = token_re.match(text, pos)
            if match:
                raw = match.group()
                if raw.startswith('/'):
                    next_binary = (yield _NameToken(pos, raw))
                elif match.group() in ('true', 'false'):
                    next_binary = (yield _BooleanToken(pos, raw))
                else:
                    try:
                        float(raw)
                        next_binary = (yield _NumberToken(pos, raw))
                    except ValueError:
                        next_binary = (yield _KeywordToken(pos, raw))
                pos = match.end()
            else:
                next_binary = (yield _DelimiterToken(pos, text[pos]))
                pos += 1

class _BalancedExpression(_Token):
    pass

def _expression(initial, tokens, data):
    if False:
        while True:
            i = 10
    '\n    Consume some number of tokens and return a balanced PostScript expression.\n\n    Parameters\n    ----------\n    initial : _Token\n        The token that triggered parsing a balanced expression.\n    tokens : iterator of _Token\n        Following tokens.\n    data : bytes\n        Underlying data that the token positions point to.\n\n    Returns\n    -------\n    _BalancedExpression\n    '
    delim_stack = []
    token = initial
    while True:
        if token.is_delim():
            if token.raw in ('[', '{'):
                delim_stack.append(token)
            elif token.raw in (']', '}'):
                if not delim_stack:
                    raise RuntimeError(f'unmatched closing token {token}')
                match = delim_stack.pop()
                if match.raw != token.opposite():
                    raise RuntimeError(f'opening token {match} closed by {token}')
                if not delim_stack:
                    break
            else:
                raise RuntimeError(f'unknown delimiter {token}')
        elif not delim_stack:
            break
        token = next(tokens)
    return _BalancedExpression(initial.pos, data[initial.pos:token.endpos()].decode('ascii', 'replace'))

class Type1Font:
    """
    A class representing a Type-1 font, for use by backends.

    Attributes
    ----------
    parts : tuple
        A 3-tuple of the cleartext part, the encrypted part, and the finale of
        zeros.

    decrypted : bytes
        The decrypted form of ``parts[1]``.

    prop : dict[str, Any]
        A dictionary of font properties. Noteworthy keys include:

        - FontName: PostScript name of the font
        - Encoding: dict from numeric codes to glyph names
        - FontMatrix: bytes object encoding a matrix
        - UniqueID: optional font identifier, dropped when modifying the font
        - CharStrings: dict from glyph names to byte code
        - Subrs: array of byte code subroutines
        - OtherSubrs: bytes object encoding some PostScript code
    """
    __slots__ = ('parts', 'decrypted', 'prop', '_pos', '_abbr')

    def __init__(self, input):
        if False:
            i = 10
            return i + 15
        '\n        Initialize a Type-1 font.\n\n        Parameters\n        ----------\n        input : str or 3-tuple\n            Either a pfb file name, or a 3-tuple of already-decoded Type-1\n            font `~.Type1Font.parts`.\n        '
        if isinstance(input, tuple) and len(input) == 3:
            self.parts = input
        else:
            with open(input, 'rb') as file:
                data = self._read(file)
            self.parts = self._split(data)
        self.decrypted = self._decrypt(self.parts[1], 'eexec')
        self._abbr = {'RD': 'RD', 'ND': 'ND', 'NP': 'NP'}
        self._parse()

    def _read(self, file):
        if False:
            return 10
        'Read the font from a file, decoding into usable parts.'
        rawdata = file.read()
        if not rawdata.startswith(b'\x80'):
            return rawdata
        data = b''
        while rawdata:
            if not rawdata.startswith(b'\x80'):
                raise RuntimeError('Broken pfb file (expected byte 128, got %d)' % rawdata[0])
            type = rawdata[1]
            if type in (1, 2):
                (length,) = struct.unpack('<i', rawdata[2:6])
                segment = rawdata[6:6 + length]
                rawdata = rawdata[6 + length:]
            if type == 1:
                data += segment
            elif type == 2:
                data += binascii.hexlify(segment)
            elif type == 3:
                break
            else:
                raise RuntimeError('Unknown segment type %d in pfb file' % type)
        return data

    def _split(self, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Split the Type 1 font into its three main parts.\n\n        The three parts are: (1) the cleartext part, which ends in a\n        eexec operator; (2) the encrypted part; (3) the fixed part,\n        which contains 512 ASCII zeros possibly divided on various\n        lines, a cleartomark operator, and possibly something else.\n        '
        idx = data.index(b'eexec')
        idx += len(b'eexec')
        while data[idx] in b' \t\r\n':
            idx += 1
        len1 = idx
        idx = data.rindex(b'cleartomark') - 1
        zeros = 512
        while zeros and data[idx] in b'0' or data[idx] in b'\r\n':
            if data[idx] in b'0':
                zeros -= 1
            idx -= 1
        if zeros:
            _log.info('Insufficiently many zeros in Type 1 font')
        idx1 = len1 + (idx - len1 + 2 & ~1)
        binary = binascii.unhexlify(data[len1:idx1])
        return (data[:len1], binary, data[idx + 1:])

    @staticmethod
    def _decrypt(ciphertext, key, ndiscard=4):
        if False:
            i = 10
            return i + 15
        '\n        Decrypt ciphertext using the Type-1 font algorithm.\n\n        The algorithm is described in Adobe\'s "Adobe Type 1 Font Format".\n        The key argument can be an integer, or one of the strings\n        \'eexec\' and \'charstring\', which map to the key specified for the\n        corresponding part of Type-1 fonts.\n\n        The ndiscard argument should be an integer, usually 4.\n        That number of bytes is discarded from the beginning of plaintext.\n        '
        key = _api.check_getitem({'eexec': 55665, 'charstring': 4330}, key=key)
        plaintext = []
        for byte in ciphertext:
            plaintext.append(byte ^ key >> 8)
            key = (key + byte) * 52845 + 22719 & 65535
        return bytes(plaintext[ndiscard:])

    @staticmethod
    def _encrypt(plaintext, key, ndiscard=4):
        if False:
            while True:
                i = 10
        '\n        Encrypt plaintext using the Type-1 font algorithm.\n\n        The algorithm is described in Adobe\'s "Adobe Type 1 Font Format".\n        The key argument can be an integer, or one of the strings\n        \'eexec\' and \'charstring\', which map to the key specified for the\n        corresponding part of Type-1 fonts.\n\n        The ndiscard argument should be an integer, usually 4. That\n        number of bytes is prepended to the plaintext before encryption.\n        This function prepends NUL bytes for reproducibility, even though\n        the original algorithm uses random bytes, presumably to avoid\n        cryptanalysis.\n        '
        key = _api.check_getitem({'eexec': 55665, 'charstring': 4330}, key=key)
        ciphertext = []
        for byte in b'\x00' * ndiscard + plaintext:
            c = byte ^ key >> 8
            ciphertext.append(c)
            key = (key + c) * 52845 + 22719 & 65535
        return bytes(ciphertext)

    def _parse(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find the values of various font properties. This limited kind\n        of parsing is described in Chapter 10 "Adobe Type Manager\n        Compatibility" of the Type-1 spec.\n        '
        prop = {'Weight': 'Regular', 'ItalicAngle': 0.0, 'isFixedPitch': False, 'UnderlinePosition': -100, 'UnderlineThickness': 50}
        pos = {}
        data = self.parts[0] + self.decrypted
        source = _tokenize(data, True)
        while True:
            try:
                token = next(source)
            except StopIteration:
                break
            if token.is_delim():
                _expression(token, source, data)
            if token.is_slash_name():
                key = token.value()
                keypos = token.pos
            else:
                continue
            if key in ('Subrs', 'CharStrings', 'Encoding', 'OtherSubrs'):
                (prop[key], endpos) = {'Subrs': self._parse_subrs, 'CharStrings': self._parse_charstrings, 'Encoding': self._parse_encoding, 'OtherSubrs': self._parse_othersubrs}[key](source, data)
                pos.setdefault(key, []).append((keypos, endpos))
                continue
            try:
                token = next(source)
            except StopIteration:
                break
            if isinstance(token, _KeywordToken):
                continue
            if token.is_delim():
                value = _expression(token, source, data).raw
            else:
                value = token.value()
            try:
                kw = next((kw for kw in source if not kw.is_keyword('readonly', 'noaccess', 'executeonly')))
            except StopIteration:
                break
            if kw.is_keyword('def', self._abbr['ND'], self._abbr['NP']):
                prop[key] = value
                pos.setdefault(key, []).append((keypos, kw.endpos()))
            if value == '{noaccess def}':
                self._abbr['ND'] = key
            elif value == '{noaccess put}':
                self._abbr['NP'] = key
            elif value == '{string currentfile exch readstring pop}':
                self._abbr['RD'] = key
        if 'FontName' not in prop:
            prop['FontName'] = prop.get('FullName') or prop.get('FamilyName') or 'Unknown'
        if 'FullName' not in prop:
            prop['FullName'] = prop['FontName']
        if 'FamilyName' not in prop:
            extras = '(?i)([ -](regular|plain|italic|oblique|(semi)?bold|(ultra)?light|extra|condensed))+$'
            prop['FamilyName'] = re.sub(extras, '', prop['FullName'])
        ndiscard = prop.get('lenIV', 4)
        cs = prop['CharStrings']
        for (key, value) in cs.items():
            cs[key] = self._decrypt(value, 'charstring', ndiscard)
        if 'Subrs' in prop:
            prop['Subrs'] = [self._decrypt(value, 'charstring', ndiscard) for value in prop['Subrs']]
        self.prop = prop
        self._pos = pos

    def _parse_subrs(self, tokens, _data):
        if False:
            i = 10
            return i + 15
        count_token = next(tokens)
        if not count_token.is_number():
            raise RuntimeError(f'Token following /Subrs must be a number, was {count_token}')
        count = count_token.value()
        array = [None] * count
        next((t for t in tokens if t.is_keyword('array')))
        for _ in range(count):
            next((t for t in tokens if t.is_keyword('dup')))
            index_token = next(tokens)
            if not index_token.is_number():
                raise RuntimeError(f'Token following dup in Subrs definition must be a number, was {index_token}')
            nbytes_token = next(tokens)
            if not nbytes_token.is_number():
                raise RuntimeError(f'Second token following dup in Subrs definition must be a number, was {nbytes_token}')
            token = next(tokens)
            if not token.is_keyword(self._abbr['RD']):
                raise RuntimeError(f"Token preceding subr must be {self._abbr['RD']}, was {token}")
            binary_token = tokens.send(1 + nbytes_token.value())
            array[index_token.value()] = binary_token.value()
        return (array, next(tokens).endpos())

    @staticmethod
    def _parse_charstrings(tokens, _data):
        if False:
            i = 10
            return i + 15
        count_token = next(tokens)
        if not count_token.is_number():
            raise RuntimeError(f'Token following /CharStrings must be a number, was {count_token}')
        count = count_token.value()
        charstrings = {}
        next((t for t in tokens if t.is_keyword('begin')))
        while True:
            token = next((t for t in tokens if t.is_keyword('end') or t.is_slash_name()))
            if token.raw == 'end':
                return (charstrings, token.endpos())
            glyphname = token.value()
            nbytes_token = next(tokens)
            if not nbytes_token.is_number():
                raise RuntimeError(f'Token following /{glyphname} in CharStrings definition must be a number, was {nbytes_token}')
            next(tokens)
            binary_token = tokens.send(1 + nbytes_token.value())
            charstrings[glyphname] = binary_token.value()

    @staticmethod
    def _parse_encoding(tokens, _data):
        if False:
            while True:
                i = 10
        encoding = {}
        while True:
            token = next((t for t in tokens if t.is_keyword('StandardEncoding', 'dup', 'def')))
            if token.is_keyword('StandardEncoding'):
                return (_StandardEncoding, token.endpos())
            if token.is_keyword('def'):
                return (encoding, token.endpos())
            index_token = next(tokens)
            if not index_token.is_number():
                _log.warning(f'Parsing encoding: expected number, got {index_token}')
                continue
            name_token = next(tokens)
            if not name_token.is_slash_name():
                _log.warning(f'Parsing encoding: expected slash-name, got {name_token}')
                continue
            encoding[index_token.value()] = name_token.value()

    @staticmethod
    def _parse_othersubrs(tokens, data):
        if False:
            for i in range(10):
                print('nop')
        init_pos = None
        while True:
            token = next(tokens)
            if init_pos is None:
                init_pos = token.pos
            if token.is_delim():
                _expression(token, tokens, data)
            elif token.is_keyword('def', 'ND', '|-'):
                return (data[init_pos:token.endpos()], token.endpos())

    def transform(self, effects):
        if False:
            i = 10
            return i + 15
        "\n        Return a new font that is slanted and/or extended.\n\n        Parameters\n        ----------\n        effects : dict\n            A dict with optional entries:\n\n            - 'slant' : float, default: 0\n                Tangent of the angle that the font is to be slanted to the\n                right. Negative values slant to the left.\n            - 'extend' : float, default: 1\n                Scaling factor for the font width. Values less than 1 condense\n                the glyphs.\n\n        Returns\n        -------\n        `Type1Font`\n        "
        fontname = self.prop['FontName']
        italicangle = self.prop['ItalicAngle']
        array = [float(x) for x in self.prop['FontMatrix'].lstrip('[').rstrip(']').split()]
        oldmatrix = np.eye(3, 3)
        oldmatrix[0:3, 0] = array[::2]
        oldmatrix[0:3, 1] = array[1::2]
        modifier = np.eye(3, 3)
        if 'slant' in effects:
            slant = effects['slant']
            fontname += f'_Slant_{int(1000 * slant)}'
            italicangle = round(float(italicangle) - np.arctan(slant) / np.pi * 180, 5)
            modifier[1, 0] = slant
        if 'extend' in effects:
            extend = effects['extend']
            fontname += f'_Extend_{int(1000 * extend)}'
            modifier[0, 0] = extend
        newmatrix = np.dot(modifier, oldmatrix)
        array[::2] = newmatrix[0:3, 0]
        array[1::2] = newmatrix[0:3, 1]
        fontmatrix = f"[{' '.join((_format_approx(x, 6) for x in array))}]"
        replacements = [(x, f'/FontName/{fontname} def') for x in self._pos['FontName']] + [(x, f'/ItalicAngle {italicangle} def') for x in self._pos['ItalicAngle']] + [(x, f'/FontMatrix {fontmatrix} readonly def') for x in self._pos['FontMatrix']] + [(x, '') for x in self._pos.get('UniqueID', [])]
        data = bytearray(self.parts[0])
        data.extend(self.decrypted)
        len0 = len(self.parts[0])
        for ((pos0, pos1), value) in sorted(replacements, reverse=True):
            data[pos0:pos1] = value.encode('ascii', 'replace')
            if pos0 < len(self.parts[0]):
                if pos1 >= len(self.parts[0]):
                    raise RuntimeError(f'text to be replaced with {value} spans the eexec boundary')
                len0 += len(value) - pos1 + pos0
        data = bytes(data)
        return Type1Font((data[:len0], self._encrypt(data[len0:], 'eexec'), self.parts[2]))
_StandardEncoding = {**{ord(letter): letter for letter in string.ascii_letters}, 0: '.notdef', 32: 'space', 33: 'exclam', 34: 'quotedbl', 35: 'numbersign', 36: 'dollar', 37: 'percent', 38: 'ampersand', 39: 'quoteright', 40: 'parenleft', 41: 'parenright', 42: 'asterisk', 43: 'plus', 44: 'comma', 45: 'hyphen', 46: 'period', 47: 'slash', 48: 'zero', 49: 'one', 50: 'two', 51: 'three', 52: 'four', 53: 'five', 54: 'six', 55: 'seven', 56: 'eight', 57: 'nine', 58: 'colon', 59: 'semicolon', 60: 'less', 61: 'equal', 62: 'greater', 63: 'question', 64: 'at', 91: 'bracketleft', 92: 'backslash', 93: 'bracketright', 94: 'asciicircum', 95: 'underscore', 96: 'quoteleft', 123: 'braceleft', 124: 'bar', 125: 'braceright', 126: 'asciitilde', 161: 'exclamdown', 162: 'cent', 163: 'sterling', 164: 'fraction', 165: 'yen', 166: 'florin', 167: 'section', 168: 'currency', 169: 'quotesingle', 170: 'quotedblleft', 171: 'guillemotleft', 172: 'guilsinglleft', 173: 'guilsinglright', 174: 'fi', 175: 'fl', 177: 'endash', 178: 'dagger', 179: 'daggerdbl', 180: 'periodcentered', 182: 'paragraph', 183: 'bullet', 184: 'quotesinglbase', 185: 'quotedblbase', 186: 'quotedblright', 187: 'guillemotright', 188: 'ellipsis', 189: 'perthousand', 191: 'questiondown', 193: 'grave', 194: 'acute', 195: 'circumflex', 196: 'tilde', 197: 'macron', 198: 'breve', 199: 'dotaccent', 200: 'dieresis', 202: 'ring', 203: 'cedilla', 205: 'hungarumlaut', 206: 'ogonek', 207: 'caron', 208: 'emdash', 225: 'AE', 227: 'ordfeminine', 232: 'Lslash', 233: 'Oslash', 234: 'OE', 235: 'ordmasculine', 241: 'ae', 245: 'dotlessi', 248: 'lslash', 249: 'oslash', 250: 'oe', 251: 'germandbls'}