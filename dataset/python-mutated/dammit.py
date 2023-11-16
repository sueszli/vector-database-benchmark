"""Beautiful Soup bonus library: Unicode, Dammit

This library converts a bytestream to Unicode through any means
necessary. It is heavily based on code from Mark Pilgrim's Universal
Feed Parser. It works best on XML and HTML, but it does not rewrite the
XML or HTML to reflect a new encoding; that's the tree builder's job.
"""
__license__ = 'MIT'
from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
chardet_module = None
try:
    import cchardet as chardet_module
except ImportError:
    try:
        import chardet as chardet_module
    except ImportError:
        try:
            import charset_normalizer as chardet_module
        except ImportError:
            chardet_module = None
if chardet_module:

    def chardet_dammit(s):
        if False:
            while True:
                i = 10
        if isinstance(s, str):
            return None
        return chardet_module.detect(s)['encoding']
else:

    def chardet_dammit(s):
        if False:
            while True:
                i = 10
        return None
xml_encoding = '^\\s*<\\?.*encoding=[\'"](.*?)[\'"].*\\?>'
html_meta = '<\\s*meta[^>]+charset\\s*=\\s*["\']?([^>]*?)[ /;\'">]'
encoding_res = dict()
encoding_res[bytes] = {'html': re.compile(html_meta.encode('ascii'), re.I), 'xml': re.compile(xml_encoding.encode('ascii'), re.I)}
encoding_res[str] = {'html': re.compile(html_meta, re.I), 'xml': re.compile(xml_encoding, re.I)}
from html.entities import html5

class EntitySubstitution(object):
    """The ability to substitute XML or HTML entities for certain characters."""

    def _populate_class_variables():
        if False:
            print('Hello World!')
        'Initialize variables used by this class to manage the plethora of\n        HTML5 named entities.\n\n        This function returns a 3-tuple containing two dictionaries\n        and a regular expression:\n\n        unicode_to_name - A mapping of Unicode strings like "⦨" to\n        entity names like "angmsdaa". When a single Unicode string has\n        multiple entity names, we try to choose the most commonly-used\n        name.\n\n        name_to_unicode: A mapping of entity names like "angmsdaa" to \n        Unicode strings like "⦨".\n\n        named_entity_re: A regular expression matching (almost) any\n        Unicode string that corresponds to an HTML5 named entity.\n        '
        unicode_to_name = {}
        name_to_unicode = {}
        short_entities = set()
        long_entities_by_first_character = defaultdict(set)
        for (name_with_semicolon, character) in sorted(html5.items()):
            if name_with_semicolon.endswith(';'):
                name = name_with_semicolon[:-1]
            else:
                name = name_with_semicolon
            if name not in name_to_unicode:
                name_to_unicode[name] = character
            unicode_to_name[character] = name
            if len(character) == 1 and ord(character) < 128 and (character not in '<>&'):
                continue
            if len(character) > 1 and all((ord(x) < 128 for x in character)):
                continue
            if len(character) == 1:
                short_entities.add(character)
            else:
                long_entities_by_first_character[character[0]].add(character)
        particles = set()
        for short in short_entities:
            long_versions = long_entities_by_first_character[short]
            if not long_versions:
                particles.add(short)
            else:
                ignore = ''.join([x[1] for x in long_versions])
                particles.add('%s(?![%s])' % (short, ignore))
        for long_entities in list(long_entities_by_first_character.values()):
            for long_entity in long_entities:
                particles.add(long_entity)
        re_definition = '(%s)' % '|'.join(particles)
        for (codepoint, name) in list(codepoint2name.items()):
            character = chr(codepoint)
            unicode_to_name[character] = name
        return (unicode_to_name, name_to_unicode, re.compile(re_definition))
    (CHARACTER_TO_HTML_ENTITY, HTML_ENTITY_TO_CHARACTER, CHARACTER_TO_HTML_ENTITY_RE) = _populate_class_variables()
    CHARACTER_TO_XML_ENTITY = {"'": 'apos', '"': 'quot', '&': 'amp', '<': 'lt', '>': 'gt'}
    BARE_AMPERSAND_OR_BRACKET = re.compile('([<>]|&(?!#\\d+;|#x[0-9a-fA-F]+;|\\w+;))')
    AMPERSAND_OR_BRACKET = re.compile('([<>&])')

    @classmethod
    def _substitute_html_entity(cls, matchobj):
        if False:
            print('Hello World!')
        'Used with a regular expression to substitute the\n        appropriate HTML entity for a special character string.'
        entity = cls.CHARACTER_TO_HTML_ENTITY.get(matchobj.group(0))
        return '&%s;' % entity

    @classmethod
    def _substitute_xml_entity(cls, matchobj):
        if False:
            i = 10
            return i + 15
        'Used with a regular expression to substitute the\n        appropriate XML entity for a special character string.'
        entity = cls.CHARACTER_TO_XML_ENTITY[matchobj.group(0)]
        return '&%s;' % entity

    @classmethod
    def quoted_attribute_value(self, value):
        if False:
            return 10
        'Make a value into a quoted XML attribute, possibly escaping it.\n\n         Most strings will be quoted using double quotes.\n\n          Bob\'s Bar -> "Bob\'s Bar"\n\n         If a string contains double quotes, it will be quoted using\n         single quotes.\n\n          Welcome to "my bar" -> \'Welcome to "my bar"\'\n\n         If a string contains both single and double quotes, the\n         double quotes will be escaped, and the string will be quoted\n         using double quotes.\n\n          Welcome to "Bob\'s Bar" -> "Welcome to &quot;Bob\'s bar&quot;\n        '
        quote_with = '"'
        if '"' in value:
            if "'" in value:
                replace_with = '&quot;'
                value = value.replace('"', replace_with)
            else:
                quote_with = "'"
        return quote_with + value + quote_with

    @classmethod
    def substitute_xml(cls, value, make_quoted_attribute=False):
        if False:
            return 10
        'Substitute XML entities for special XML characters.\n\n        :param value: A string to be substituted. The less-than sign\n          will become &lt;, the greater-than sign will become &gt;,\n          and any ampersands will become &amp;. If you want ampersands\n          that appear to be part of an entity definition to be left\n          alone, use substitute_xml_containing_entities() instead.\n\n        :param make_quoted_attribute: If True, then the string will be\n         quoted, as befits an attribute value.\n        '
        value = cls.AMPERSAND_OR_BRACKET.sub(cls._substitute_xml_entity, value)
        if make_quoted_attribute:
            value = cls.quoted_attribute_value(value)
        return value

    @classmethod
    def substitute_xml_containing_entities(cls, value, make_quoted_attribute=False):
        if False:
            while True:
                i = 10
        'Substitute XML entities for special XML characters.\n\n        :param value: A string to be substituted. The less-than sign will\n          become &lt;, the greater-than sign will become &gt;, and any\n          ampersands that are not part of an entity defition will\n          become &amp;.\n\n        :param make_quoted_attribute: If True, then the string will be\n         quoted, as befits an attribute value.\n        '
        value = cls.BARE_AMPERSAND_OR_BRACKET.sub(cls._substitute_xml_entity, value)
        if make_quoted_attribute:
            value = cls.quoted_attribute_value(value)
        return value

    @classmethod
    def substitute_html(cls, s):
        if False:
            return 10
        'Replace certain Unicode characters with named HTML entities.\n\n        This differs from data.encode(encoding, \'xmlcharrefreplace\')\n        in that the goal is to make the result more readable (to those\n        with ASCII displays) rather than to recover from\n        errors. There\'s absolutely nothing wrong with a UTF-8 string\n        containg a LATIN SMALL LETTER E WITH ACUTE, but replacing that\n        character with "&eacute;" will make it more readable to some\n        people.\n\n        :param s: A Unicode string.\n        '
        return cls.CHARACTER_TO_HTML_ENTITY_RE.sub(cls._substitute_html_entity, s)

class EncodingDetector:
    """Suggests a number of possible encodings for a bytestring.

    Order of precedence:

    1. Encodings you specifically tell EncodingDetector to try first
    (the known_definite_encodings argument to the constructor).

    2. An encoding determined by sniffing the document's byte-order mark.

    3. Encodings you specifically tell EncodingDetector to try if
    byte-order mark sniffing fails (the user_encodings argument to the
    constructor).

    4. An encoding declared within the bytestring itself, either in an
    XML declaration (if the bytestring is to be interpreted as an XML
    document), or in a <meta> tag (if the bytestring is to be
    interpreted as an HTML document.)

    5. An encoding detected through textual analysis by chardet,
    cchardet, or a similar external library.

    4. UTF-8.

    5. Windows-1252.

    """

    def __init__(self, markup, known_definite_encodings=None, is_html=False, exclude_encodings=None, user_encodings=None, override_encodings=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructor.\n\n        :param markup: Some markup in an unknown encoding.\n\n        :param known_definite_encodings: When determining the encoding\n            of `markup`, these encodings will be tried first, in\n            order. In HTML terms, this corresponds to the "known\n            definite encoding" step defined here:\n            https://html.spec.whatwg.org/multipage/parsing.html#parsing-with-a-known-character-encoding\n\n        :param user_encodings: These encodings will be tried after the\n            `known_definite_encodings` have been tried and failed, and\n            after an attempt to sniff the encoding by looking at a\n            byte order mark has failed. In HTML terms, this\n            corresponds to the step "user has explicitly instructed\n            the user agent to override the document\'s character\n            encoding", defined here:\n            https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding\n\n        :param override_encodings: A deprecated alias for\n            known_definite_encodings. Any encodings here will be tried\n            immediately after the encodings in\n            known_definite_encodings.\n\n        :param is_html: If True, this markup is considered to be\n            HTML. Otherwise it\'s assumed to be XML.\n\n        :param exclude_encodings: These encodings will not be tried,\n            even if they otherwise would be.\n\n        '
        self.known_definite_encodings = list(known_definite_encodings or [])
        if override_encodings:
            self.known_definite_encodings += override_encodings
        self.user_encodings = user_encodings or []
        exclude_encodings = exclude_encodings or []
        self.exclude_encodings = set([x.lower() for x in exclude_encodings])
        self.chardet_encoding = None
        self.is_html = is_html
        self.declared_encoding = None
        (self.markup, self.sniffed_encoding) = self.strip_byte_order_mark(markup)

    def _usable(self, encoding, tried):
        if False:
            while True:
                i = 10
        'Should we even bother to try this encoding?\n\n        :param encoding: Name of an encoding.\n        :param tried: Encodings that have already been tried. This will be modified\n            as a side effect.\n        '
        if encoding is not None:
            encoding = encoding.lower()
            if encoding in self.exclude_encodings:
                return False
            if encoding not in tried:
                tried.add(encoding)
                return True
        return False

    @property
    def encodings(self):
        if False:
            while True:
                i = 10
        'Yield a number of encodings that might work for this markup.\n\n        :yield: A sequence of strings.\n        '
        tried = set()
        for e in self.known_definite_encodings:
            if self._usable(e, tried):
                yield e
        if self._usable(self.sniffed_encoding, tried):
            yield self.sniffed_encoding
        for e in self.user_encodings:
            if self._usable(e, tried):
                yield e
        if self.declared_encoding is None:
            self.declared_encoding = self.find_declared_encoding(self.markup, self.is_html)
        if self._usable(self.declared_encoding, tried):
            yield self.declared_encoding
        if self.chardet_encoding is None:
            self.chardet_encoding = chardet_dammit(self.markup)
        if self._usable(self.chardet_encoding, tried):
            yield self.chardet_encoding
        for e in ('utf-8', 'windows-1252'):
            if self._usable(e, tried):
                yield e

    @classmethod
    def strip_byte_order_mark(cls, data):
        if False:
            i = 10
            return i + 15
        'If a byte-order mark is present, strip it and return the encoding it implies.\n\n        :param data: Some markup.\n        :return: A 2-tuple (modified data, implied encoding)\n        '
        encoding = None
        if isinstance(data, str):
            return (data, encoding)
        if len(data) >= 4 and data[:2] == b'\xfe\xff' and (data[2:4] != '\x00\x00'):
            encoding = 'utf-16be'
            data = data[2:]
        elif len(data) >= 4 and data[:2] == b'\xff\xfe' and (data[2:4] != '\x00\x00'):
            encoding = 'utf-16le'
            data = data[2:]
        elif data[:3] == b'\xef\xbb\xbf':
            encoding = 'utf-8'
            data = data[3:]
        elif data[:4] == b'\x00\x00\xfe\xff':
            encoding = 'utf-32be'
            data = data[4:]
        elif data[:4] == b'\xff\xfe\x00\x00':
            encoding = 'utf-32le'
            data = data[4:]
        return (data, encoding)

    @classmethod
    def find_declared_encoding(cls, markup, is_html=False, search_entire_document=False):
        if False:
            for i in range(10):
                print('nop')
        "Given a document, tries to find its declared encoding.\n\n        An XML encoding is declared at the beginning of the document.\n\n        An HTML encoding is declared in a <meta> tag, hopefully near the\n        beginning of the document.\n\n        :param markup: Some markup.\n        :param is_html: If True, this markup is considered to be HTML. Otherwise\n            it's assumed to be XML.\n        :param search_entire_document: Since an encoding is supposed to declared near the beginning\n            of the document, most of the time it's only necessary to search a few kilobytes of data.\n            Set this to True to force this method to search the entire document.\n        "
        if search_entire_document:
            xml_endpos = html_endpos = len(markup)
        else:
            xml_endpos = 1024
            html_endpos = max(2048, int(len(markup) * 0.05))
        if isinstance(markup, bytes):
            res = encoding_res[bytes]
        else:
            res = encoding_res[str]
        xml_re = res['xml']
        html_re = res['html']
        declared_encoding = None
        declared_encoding_match = xml_re.search(markup, endpos=xml_endpos)
        if not declared_encoding_match and is_html:
            declared_encoding_match = html_re.search(markup, endpos=html_endpos)
        if declared_encoding_match is not None:
            declared_encoding = declared_encoding_match.groups()[0]
        if declared_encoding:
            if isinstance(declared_encoding, bytes):
                declared_encoding = declared_encoding.decode('ascii', 'replace')
            return declared_encoding.lower()
        return None

class UnicodeDammit:
    """A class for detecting the encoding of a *ML document and
    converting it to a Unicode string. If the source encoding is
    windows-1252, can replace MS smart quotes with their HTML or XML
    equivalents."""
    CHARSET_ALIASES = {'macintosh': 'mac-roman', 'x-sjis': 'shift-jis'}
    ENCODINGS_WITH_SMART_QUOTES = ['windows-1252', 'iso-8859-1', 'iso-8859-2']

    def __init__(self, markup, known_definite_encodings=[], smart_quotes_to=None, is_html=False, exclude_encodings=[], user_encodings=None, override_encodings=None):
        if False:
            while True:
                i = 10
        'Constructor.\n\n        :param markup: A bytestring representing markup in an unknown encoding.\n\n        :param known_definite_encodings: When determining the encoding\n            of `markup`, these encodings will be tried first, in\n            order. In HTML terms, this corresponds to the "known\n            definite encoding" step defined here:\n            https://html.spec.whatwg.org/multipage/parsing.html#parsing-with-a-known-character-encoding\n\n        :param user_encodings: These encodings will be tried after the\n            `known_definite_encodings` have been tried and failed, and\n            after an attempt to sniff the encoding by looking at a\n            byte order mark has failed. In HTML terms, this\n            corresponds to the step "user has explicitly instructed\n            the user agent to override the document\'s character\n            encoding", defined here:\n            https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding\n\n        :param override_encodings: A deprecated alias for\n            known_definite_encodings. Any encodings here will be tried\n            immediately after the encodings in\n            known_definite_encodings.\n\n        :param smart_quotes_to: By default, Microsoft smart quotes will, like all other characters, be converted\n           to Unicode characters. Setting this to \'ascii\' will convert them to ASCII quotes instead.\n           Setting it to \'xml\' will convert them to XML entity references, and setting it to \'html\'\n           will convert them to HTML entity references.\n        :param is_html: If True, this markup is considered to be HTML. Otherwise\n            it\'s assumed to be XML.\n        :param exclude_encodings: These encodings will not be considered, even\n            if the sniffing code thinks they might make sense.\n\n        '
        self.smart_quotes_to = smart_quotes_to
        self.tried_encodings = []
        self.contains_replacement_characters = False
        self.is_html = is_html
        self.log = logging.getLogger(__name__)
        self.detector = EncodingDetector(markup, known_definite_encodings, is_html, exclude_encodings, user_encodings, override_encodings)
        if isinstance(markup, str) or markup == '':
            self.markup = markup
            self.unicode_markup = str(markup)
            self.original_encoding = None
            return
        self.markup = self.detector.markup
        u = None
        for encoding in self.detector.encodings:
            markup = self.detector.markup
            u = self._convert_from(encoding)
            if u is not None:
                break
        if not u:
            for encoding in self.detector.encodings:
                if encoding != 'ascii':
                    u = self._convert_from(encoding, 'replace')
                if u is not None:
                    self.log.warning('Some characters could not be decoded, and were replaced with REPLACEMENT CHARACTER.')
                    self.contains_replacement_characters = True
                    break
        self.unicode_markup = u
        if not u:
            self.original_encoding = None

    def _sub_ms_char(self, match):
        if False:
            while True:
                i = 10
        'Changes a MS smart quote character to an XML or HTML\n        entity, or an ASCII character.'
        orig = match.group(1)
        if self.smart_quotes_to == 'ascii':
            sub = self.MS_CHARS_TO_ASCII.get(orig).encode()
        else:
            sub = self.MS_CHARS.get(orig)
            if type(sub) == tuple:
                if self.smart_quotes_to == 'xml':
                    sub = '&#x'.encode() + sub[1].encode() + ';'.encode()
                else:
                    sub = '&'.encode() + sub[0].encode() + ';'.encode()
            else:
                sub = sub.encode()
        return sub

    def _convert_from(self, proposed, errors='strict'):
        if False:
            i = 10
            return i + 15
        'Attempt to convert the markup to the proposed encoding.\n\n        :param proposed: The name of a character encoding.\n        '
        proposed = self.find_codec(proposed)
        if not proposed or (proposed, errors) in self.tried_encodings:
            return None
        self.tried_encodings.append((proposed, errors))
        markup = self.markup
        if self.smart_quotes_to is not None and proposed in self.ENCODINGS_WITH_SMART_QUOTES:
            smart_quotes_re = b'([\x80-\x9f])'
            smart_quotes_compiled = re.compile(smart_quotes_re)
            markup = smart_quotes_compiled.sub(self._sub_ms_char, markup)
        try:
            u = self._to_unicode(markup, proposed, errors)
            self.markup = u
            self.original_encoding = proposed
        except Exception as e:
            return None
        return self.markup

    def _to_unicode(self, data, encoding, errors='strict'):
        if False:
            for i in range(10):
                print('nop')
        'Given a string and its encoding, decodes the string into Unicode.\n\n        :param encoding: The name of an encoding.\n        '
        return str(data, encoding, errors)

    @property
    def declared_html_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        'If the markup is an HTML document, returns the encoding declared _within_\n        the document.\n        '
        if not self.is_html:
            return None
        return self.detector.declared_encoding

    def find_codec(self, charset):
        if False:
            i = 10
            return i + 15
        'Convert the name of a character set to a codec name.\n\n        :param charset: The name of a character set.\n        :return: The name of a codec.\n        '
        value = self._codec(self.CHARSET_ALIASES.get(charset, charset)) or (charset and self._codec(charset.replace('-', ''))) or (charset and self._codec(charset.replace('-', '_'))) or (charset and charset.lower()) or charset
        if value:
            return value.lower()
        return None

    def _codec(self, charset):
        if False:
            return 10
        if not charset:
            return charset
        codec = None
        try:
            codecs.lookup(charset)
            codec = charset
        except (LookupError, ValueError):
            pass
        return codec
    MS_CHARS = {b'\x80': ('euro', '20AC'), b'\x81': ' ', b'\x82': ('sbquo', '201A'), b'\x83': ('fnof', '192'), b'\x84': ('bdquo', '201E'), b'\x85': ('hellip', '2026'), b'\x86': ('dagger', '2020'), b'\x87': ('Dagger', '2021'), b'\x88': ('circ', '2C6'), b'\x89': ('permil', '2030'), b'\x8a': ('Scaron', '160'), b'\x8b': ('lsaquo', '2039'), b'\x8c': ('OElig', '152'), b'\x8d': '?', b'\x8e': ('#x17D', '17D'), b'\x8f': '?', b'\x90': '?', b'\x91': ('lsquo', '2018'), b'\x92': ('rsquo', '2019'), b'\x93': ('ldquo', '201C'), b'\x94': ('rdquo', '201D'), b'\x95': ('bull', '2022'), b'\x96': ('ndash', '2013'), b'\x97': ('mdash', '2014'), b'\x98': ('tilde', '2DC'), b'\x99': ('trade', '2122'), b'\x9a': ('scaron', '161'), b'\x9b': ('rsaquo', '203A'), b'\x9c': ('oelig', '153'), b'\x9d': '?', b'\x9e': ('#x17E', '17E'), b'\x9f': ('Yuml', '')}
    MS_CHARS_TO_ASCII = {b'\x80': 'EUR', b'\x81': ' ', b'\x82': ',', b'\x83': 'f', b'\x84': ',,', b'\x85': '...', b'\x86': '+', b'\x87': '++', b'\x88': '^', b'\x89': '%', b'\x8a': 'S', b'\x8b': '<', b'\x8c': 'OE', b'\x8d': '?', b'\x8e': 'Z', b'\x8f': '?', b'\x90': '?', b'\x91': "'", b'\x92': "'", b'\x93': '"', b'\x94': '"', b'\x95': '*', b'\x96': '-', b'\x97': '--', b'\x98': '~', b'\x99': '(TM)', b'\x9a': 's', b'\x9b': '>', b'\x9c': 'oe', b'\x9d': '?', b'\x9e': 'z', b'\x9f': 'Y', b'\xa0': ' ', b'\xa1': '!', b'\xa2': 'c', b'\xa3': 'GBP', b'\xa4': '$', b'\xa5': 'YEN', b'\xa6': '|', b'\xa7': 'S', b'\xa8': '..', b'\xa9': '', b'\xaa': '(th)', b'\xab': '<<', b'\xac': '!', b'\xad': ' ', b'\xae': '(R)', b'\xaf': '-', b'\xb0': 'o', b'\xb1': '+-', b'\xb2': '2', b'\xb3': '3', b'\xb4': ("'", 'acute'), b'\xb5': 'u', b'\xb6': 'P', b'\xb7': '*', b'\xb8': ',', b'\xb9': '1', b'\xba': '(th)', b'\xbb': '>>', b'\xbc': '1/4', b'\xbd': '1/2', b'\xbe': '3/4', b'\xbf': '?', b'\xc0': 'A', b'\xc1': 'A', b'\xc2': 'A', b'\xc3': 'A', b'\xc4': 'A', b'\xc5': 'A', b'\xc6': 'AE', b'\xc7': 'C', b'\xc8': 'E', b'\xc9': 'E', b'\xca': 'E', b'\xcb': 'E', b'\xcc': 'I', b'\xcd': 'I', b'\xce': 'I', b'\xcf': 'I', b'\xd0': 'D', b'\xd1': 'N', b'\xd2': 'O', b'\xd3': 'O', b'\xd4': 'O', b'\xd5': 'O', b'\xd6': 'O', b'\xd7': '*', b'\xd8': 'O', b'\xd9': 'U', b'\xda': 'U', b'\xdb': 'U', b'\xdc': 'U', b'\xdd': 'Y', b'\xde': 'b', b'\xdf': 'B', b'\xe0': 'a', b'\xe1': 'a', b'\xe2': 'a', b'\xe3': 'a', b'\xe4': 'a', b'\xe5': 'a', b'\xe6': 'ae', b'\xe7': 'c', b'\xe8': 'e', b'\xe9': 'e', b'\xea': 'e', b'\xeb': 'e', b'\xec': 'i', b'\xed': 'i', b'\xee': 'i', b'\xef': 'i', b'\xf0': 'o', b'\xf1': 'n', b'\xf2': 'o', b'\xf3': 'o', b'\xf4': 'o', b'\xf5': 'o', b'\xf6': 'o', b'\xf7': '/', b'\xf8': 'o', b'\xf9': 'u', b'\xfa': 'u', b'\xfb': 'u', b'\xfc': 'u', b'\xfd': 'y', b'\xfe': 'b', b'\xff': 'y'}
    WINDOWS_1252_TO_UTF8 = {128: b'\xe2\x82\xac', 130: b'\xe2\x80\x9a', 131: b'\xc6\x92', 132: b'\xe2\x80\x9e', 133: b'\xe2\x80\xa6', 134: b'\xe2\x80\xa0', 135: b'\xe2\x80\xa1', 136: b'\xcb\x86', 137: b'\xe2\x80\xb0', 138: b'\xc5\xa0', 139: b'\xe2\x80\xb9', 140: b'\xc5\x92', 142: b'\xc5\xbd', 145: b'\xe2\x80\x98', 146: b'\xe2\x80\x99', 147: b'\xe2\x80\x9c', 148: b'\xe2\x80\x9d', 149: b'\xe2\x80\xa2', 150: b'\xe2\x80\x93', 151: b'\xe2\x80\x94', 152: b'\xcb\x9c', 153: b'\xe2\x84\xa2', 154: b'\xc5\xa1', 155: b'\xe2\x80\xba', 156: b'\xc5\x93', 158: b'\xc5\xbe', 159: b'\xc5\xb8', 160: b'\xc2\xa0', 161: b'\xc2\xa1', 162: b'\xc2\xa2', 163: b'\xc2\xa3', 164: b'\xc2\xa4', 165: b'\xc2\xa5', 166: b'\xc2\xa6', 167: b'\xc2\xa7', 168: b'\xc2\xa8', 169: b'\xc2\xa9', 170: b'\xc2\xaa', 171: b'\xc2\xab', 172: b'\xc2\xac', 173: b'\xc2\xad', 174: b'\xc2\xae', 175: b'\xc2\xaf', 176: b'\xc2\xb0', 177: b'\xc2\xb1', 178: b'\xc2\xb2', 179: b'\xc2\xb3', 180: b'\xc2\xb4', 181: b'\xc2\xb5', 182: b'\xc2\xb6', 183: b'\xc2\xb7', 184: b'\xc2\xb8', 185: b'\xc2\xb9', 186: b'\xc2\xba', 187: b'\xc2\xbb', 188: b'\xc2\xbc', 189: b'\xc2\xbd', 190: b'\xc2\xbe', 191: b'\xc2\xbf', 192: b'\xc3\x80', 193: b'\xc3\x81', 194: b'\xc3\x82', 195: b'\xc3\x83', 196: b'\xc3\x84', 197: b'\xc3\x85', 198: b'\xc3\x86', 199: b'\xc3\x87', 200: b'\xc3\x88', 201: b'\xc3\x89', 202: b'\xc3\x8a', 203: b'\xc3\x8b', 204: b'\xc3\x8c', 205: b'\xc3\x8d', 206: b'\xc3\x8e', 207: b'\xc3\x8f', 208: b'\xc3\x90', 209: b'\xc3\x91', 210: b'\xc3\x92', 211: b'\xc3\x93', 212: b'\xc3\x94', 213: b'\xc3\x95', 214: b'\xc3\x96', 215: b'\xc3\x97', 216: b'\xc3\x98', 217: b'\xc3\x99', 218: b'\xc3\x9a', 219: b'\xc3\x9b', 220: b'\xc3\x9c', 221: b'\xc3\x9d', 222: b'\xc3\x9e', 223: b'\xc3\x9f', 224: b'\xc3\xa0', 225: b'\xa1', 226: b'\xc3\xa2', 227: b'\xc3\xa3', 228: b'\xc3\xa4', 229: b'\xc3\xa5', 230: b'\xc3\xa6', 231: b'\xc3\xa7', 232: b'\xc3\xa8', 233: b'\xc3\xa9', 234: b'\xc3\xaa', 235: b'\xc3\xab', 236: b'\xc3\xac', 237: b'\xc3\xad', 238: b'\xc3\xae', 239: b'\xc3\xaf', 240: b'\xc3\xb0', 241: b'\xc3\xb1', 242: b'\xc3\xb2', 243: b'\xc3\xb3', 244: b'\xc3\xb4', 245: b'\xc3\xb5', 246: b'\xc3\xb6', 247: b'\xc3\xb7', 248: b'\xc3\xb8', 249: b'\xc3\xb9', 250: b'\xc3\xba', 251: b'\xc3\xbb', 252: b'\xc3\xbc', 253: b'\xc3\xbd', 254: b'\xc3\xbe'}
    MULTIBYTE_MARKERS_AND_SIZES = [(194, 223, 2), (224, 239, 3), (240, 244, 4)]
    FIRST_MULTIBYTE_MARKER = MULTIBYTE_MARKERS_AND_SIZES[0][0]
    LAST_MULTIBYTE_MARKER = MULTIBYTE_MARKERS_AND_SIZES[-1][1]

    @classmethod
    def detwingle(cls, in_bytes, main_encoding='utf8', embedded_encoding='windows-1252'):
        if False:
            i = 10
            return i + 15
        "Fix characters from one encoding embedded in some other encoding.\n\n        Currently the only situation supported is Windows-1252 (or its\n        subset ISO-8859-1), embedded in UTF-8.\n\n        :param in_bytes: A bytestring that you suspect contains\n            characters from multiple encodings. Note that this _must_\n            be a bytestring. If you've already converted the document\n            to Unicode, you're too late.\n        :param main_encoding: The primary encoding of `in_bytes`.\n        :param embedded_encoding: The encoding that was used to embed characters\n            in the main document.\n        :return: A bytestring in which `embedded_encoding`\n          characters have been converted to their `main_encoding`\n          equivalents.\n        "
        if embedded_encoding.replace('_', '-').lower() not in ('windows-1252', 'windows_1252'):
            raise NotImplementedError('Windows-1252 and ISO-8859-1 are the only currently supported embedded encodings.')
        if main_encoding.lower() not in ('utf8', 'utf-8'):
            raise NotImplementedError('UTF-8 is the only currently supported main encoding.')
        byte_chunks = []
        chunk_start = 0
        pos = 0
        while pos < len(in_bytes):
            byte = in_bytes[pos]
            if not isinstance(byte, int):
                byte = ord(byte)
            if byte >= cls.FIRST_MULTIBYTE_MARKER and byte <= cls.LAST_MULTIBYTE_MARKER:
                for (start, end, size) in cls.MULTIBYTE_MARKERS_AND_SIZES:
                    if byte >= start and byte <= end:
                        pos += size
                        break
            elif byte >= 128 and byte in cls.WINDOWS_1252_TO_UTF8:
                byte_chunks.append(in_bytes[chunk_start:pos])
                byte_chunks.append(cls.WINDOWS_1252_TO_UTF8[byte])
                pos += 1
                chunk_start = pos
            else:
                pos += 1
        if chunk_start == 0:
            return in_bytes
        else:
            byte_chunks.append(in_bytes[chunk_start:])
        return b''.join(byte_chunks)