"""So much practical programming involves string manipulation, which
Python readily accommodates. Still, there are dozens of basic and
common capabilities missing from the standard library, several of them
provided by ``strutils``.
"""
from __future__ import print_function
import re
import sys
import uuid
import zlib
import string
import unicodedata
import collections
from gzip import GzipFile
try:
    from cStringIO import cStringIO as StringIO
except ImportError:
    from io import BytesIO as StringIO
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
try:
    (unicode, str, bytes, basestring) = (unicode, str, str, basestring)
    from HTMLParser import HTMLParser
    import htmlentitydefs
except NameError:
    (unicode, str, bytes, basestring) = (str, bytes, bytes, (str, bytes))
    unichr = chr
    from html.parser import HTMLParser
    from html import entities as htmlentitydefs
try:
    import __builtin__ as builtins
except ImportError:
    import builtins
__all__ = ['camel2under', 'under2camel', 'slugify', 'split_punct_ws', 'unit_len', 'ordinalize', 'cardinalize', 'pluralize', 'singularize', 'asciify', 'is_ascii', 'is_uuid', 'html2text', 'strip_ansi', 'bytes2human', 'find_hashtags', 'a10n', 'gzip_bytes', 'gunzip_bytes', 'iter_splitlines', 'indent', 'escape_shell_args', 'args2cmd', 'args2sh', 'parse_int_list', 'format_int_list', 'int_list_complement', 'int_list_to_int_tuples', 'MultiReplace', 'multi_replace', 'unwrap_text']
_punct_ws_str = string.punctuation + string.whitespace
_punct_re = re.compile('[' + _punct_ws_str + ']+')
_camel2under_re = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')

def camel2under(camel_string):
    if False:
        i = 10
        return i + 15
    "Converts a camelcased string to underscores. Useful for turning a\n    class name into a function name.\n\n    >>> camel2under('BasicParseTest')\n    'basic_parse_test'\n    "
    return _camel2under_re.sub('_\\1', camel_string).lower()

def under2camel(under_string):
    if False:
        i = 10
        return i + 15
    "Converts an underscored string to camelcased. Useful for turning a\n    function name into a class name.\n\n    >>> under2camel('complex_tokenizer')\n    'ComplexTokenizer'\n    "
    return ''.join((w.capitalize() or '_' for w in under_string.split('_')))

def slugify(text, delim='_', lower=True, ascii=False):
    if False:
        print('Hello World!')
    '\n    A basic function that turns text full of scary characters\n    (i.e., punctuation and whitespace), into a relatively safe\n    lowercased string separated only by the delimiter specified\n    by *delim*, which defaults to ``_``.\n\n    The *ascii* convenience flag will :func:`asciify` the slug if\n    you require ascii-only slugs.\n\n    >>> slugify(\'First post! Hi!!!!~1    \')\n    \'first_post_hi_1\'\n\n    >>> slugify("Kurt Gödel\'s pretty cool.", ascii=True) ==         b\'kurt_goedel_s_pretty_cool\'\n    True\n\n    '
    ret = delim.join(split_punct_ws(text)) or delim if text else ''
    if ascii:
        ret = asciify(ret)
    if lower:
        ret = ret.lower()
    return ret

def split_punct_ws(text):
    if False:
        for i in range(10):
            print('nop')
    "While :meth:`str.split` will split on whitespace,\n    :func:`split_punct_ws` will split on punctuation and\n    whitespace. This used internally by :func:`slugify`, above.\n\n    >>> split_punct_ws('First post! Hi!!!!~1    ')\n    ['First', 'post', 'Hi', '1']\n    "
    return [w for w in _punct_re.split(text) if w]

def unit_len(sized_iterable, unit_noun='item'):
    if False:
        for i in range(10):
            print('nop')
    "Returns a plain-English description of an iterable's\n    :func:`len()`, conditionally pluralized with :func:`cardinalize`,\n    detailed below.\n\n    >>> print(unit_len(range(10), 'number'))\n    10 numbers\n    >>> print(unit_len('aeiou', 'vowel'))\n    5 vowels\n    >>> print(unit_len([], 'worry'))\n    No worries\n    "
    count = len(sized_iterable)
    units = cardinalize(unit_noun, count)
    if count:
        return u'%s %s' % (count, units)
    return u'No %s' % (units,)
_ORDINAL_MAP = {'1': 'st', '2': 'nd', '3': 'rd'}

def ordinalize(number, ext_only=False):
    if False:
        return 10
    "Turns *number* into its cardinal form, i.e., 1st, 2nd,\n    3rd, 4th, etc. If the last character isn't a digit, it returns the\n    string value unchanged.\n\n    Args:\n        number (int or str): Number to be cardinalized.\n        ext_only (bool): Whether to return only the suffix. Default ``False``.\n\n    >>> print(ordinalize(1))\n    1st\n    >>> print(ordinalize(3694839230))\n    3694839230th\n    >>> print(ordinalize('hi'))\n    hi\n    >>> print(ordinalize(1515))\n    1515th\n    "
    (numstr, ext) = (unicode(number), '')
    if numstr and numstr[-1] in string.digits:
        try:
            if numstr[-2] == '1':
                ext = 'th'
            else:
                ext = _ORDINAL_MAP.get(numstr[-1], 'th')
        except IndexError:
            ext = _ORDINAL_MAP.get(numstr[-1], 'th')
    if ext_only:
        return ext
    else:
        return numstr + ext

def cardinalize(unit_noun, count):
    if False:
        for i in range(10):
            print('nop')
    "Conditionally pluralizes a singular word *unit_noun* if\n    *count* is not one, preserving case when possible.\n\n    >>> vowels = 'aeiou'\n    >>> print(len(vowels), cardinalize('vowel', len(vowels)))\n    5 vowels\n    >>> print(3, cardinalize('Wish', 3))\n    3 Wishes\n    "
    if count == 1:
        return unit_noun
    return pluralize(unit_noun)

def singularize(word):
    if False:
        for i in range(10):
            print('nop')
    "Semi-intelligently converts an English plural *word* to its\n    singular form, preserving case pattern.\n\n    >>> singularize('chances')\n    'chance'\n    >>> singularize('Activities')\n    'Activity'\n    >>> singularize('Glasses')\n    'Glass'\n    >>> singularize('FEET')\n    'FOOT'\n\n    "
    (orig_word, word) = (word, word.strip().lower())
    if not word or word in _IRR_S2P:
        return orig_word
    irr_singular = _IRR_P2S.get(word)
    if irr_singular:
        singular = irr_singular
    elif not word.endswith('s'):
        return orig_word
    elif len(word) == 2:
        singular = word[:-1]
    elif word.endswith('ies') and word[-4:-3] not in 'aeiou':
        singular = word[:-3] + 'y'
    elif word.endswith('es') and word[-3] == 's':
        singular = word[:-2]
    else:
        singular = word[:-1]
    return _match_case(orig_word, singular)

def pluralize(word):
    if False:
        for i in range(10):
            print('nop')
    "Semi-intelligently converts an English *word* from singular form to\n    plural, preserving case pattern.\n\n    >>> pluralize('friend')\n    'friends'\n    >>> pluralize('enemy')\n    'enemies'\n    >>> pluralize('Sheep')\n    'Sheep'\n    "
    (orig_word, word) = (word, word.strip().lower())
    if not word or word in _IRR_P2S:
        return orig_word
    irr_plural = _IRR_S2P.get(word)
    if irr_plural:
        plural = irr_plural
    elif word.endswith('y') and word[-2:-1] not in 'aeiou':
        plural = word[:-1] + 'ies'
    elif word[-1] == 's' or word.endswith('ch') or word.endswith('sh'):
        plural = word if word.endswith('es') else word + 'es'
    else:
        plural = word + 's'
    return _match_case(orig_word, plural)

def _match_case(master, disciple):
    if False:
        i = 10
        return i + 15
    if not master.strip():
        return disciple
    if master.lower() == master:
        return disciple.lower()
    elif master.upper() == master:
        return disciple.upper()
    elif master.title() == master:
        return disciple.title()
    return disciple
_IRR_S2P = {'addendum': 'addenda', 'alga': 'algae', 'alumna': 'alumnae', 'alumnus': 'alumni', 'analysis': 'analyses', 'antenna': 'antennae', 'appendix': 'appendices', 'axis': 'axes', 'bacillus': 'bacilli', 'bacterium': 'bacteria', 'basis': 'bases', 'beau': 'beaux', 'bison': 'bison', 'bureau': 'bureaus', 'cactus': 'cacti', 'calf': 'calves', 'child': 'children', 'corps': 'corps', 'corpus': 'corpora', 'crisis': 'crises', 'criterion': 'criteria', 'curriculum': 'curricula', 'datum': 'data', 'deer': 'deer', 'diagnosis': 'diagnoses', 'die': 'dice', 'dwarf': 'dwarves', 'echo': 'echoes', 'elf': 'elves', 'ellipsis': 'ellipses', 'embargo': 'embargoes', 'emphasis': 'emphases', 'erratum': 'errata', 'fireman': 'firemen', 'fish': 'fish', 'focus': 'foci', 'foot': 'feet', 'formula': 'formulae', 'formula': 'formulas', 'fungus': 'fungi', 'genus': 'genera', 'goose': 'geese', 'half': 'halves', 'hero': 'heroes', 'hippopotamus': 'hippopotami', 'hoof': 'hooves', 'hypothesis': 'hypotheses', 'index': 'indices', 'knife': 'knives', 'leaf': 'leaves', 'life': 'lives', 'loaf': 'loaves', 'louse': 'lice', 'man': 'men', 'matrix': 'matrices', 'means': 'means', 'medium': 'media', 'memorandum': 'memoranda', 'millennium': 'milennia', 'moose': 'moose', 'mosquito': 'mosquitoes', 'mouse': 'mice', 'nebula': 'nebulae', 'neurosis': 'neuroses', 'nucleus': 'nuclei', 'oasis': 'oases', 'octopus': 'octopi', 'offspring': 'offspring', 'ovum': 'ova', 'ox': 'oxen', 'paralysis': 'paralyses', 'parenthesis': 'parentheses', 'person': 'people', 'phenomenon': 'phenomena', 'potato': 'potatoes', 'radius': 'radii', 'scarf': 'scarves', 'scissors': 'scissors', 'self': 'selves', 'sense': 'senses', 'series': 'series', 'sheep': 'sheep', 'shelf': 'shelves', 'species': 'species', 'stimulus': 'stimuli', 'stratum': 'strata', 'syllabus': 'syllabi', 'symposium': 'symposia', 'synopsis': 'synopses', 'synthesis': 'syntheses', 'tableau': 'tableaux', 'that': 'those', 'thesis': 'theses', 'thief': 'thieves', 'this': 'these', 'tomato': 'tomatoes', 'tooth': 'teeth', 'torpedo': 'torpedoes', 'vertebra': 'vertebrae', 'veto': 'vetoes', 'vita': 'vitae', 'watch': 'watches', 'wife': 'wives', 'wolf': 'wolves', 'woman': 'women'}
_IRR_P2S = dict([(v, k) for (k, v) in _IRR_S2P.items()])
HASHTAG_RE = re.compile('(?:^|\\s)[＃#]{1}(\\w+)', re.UNICODE)

def find_hashtags(string):
    if False:
        i = 10
        return i + 15
    "Finds and returns all hashtags in a string, with the hashmark\n    removed. Supports full-width hashmarks for Asian languages and\n    does not false-positive on URL anchors.\n\n    >>> find_hashtags('#atag http://asite/#ananchor')\n    ['atag']\n\n    ``find_hashtags`` also works with unicode hashtags.\n    "
    return HASHTAG_RE.findall(string)

def a10n(string):
    if False:
        while True:
            i = 10
    'That thing where "internationalization" becomes "i18n", what\'s it\n    called? Abbreviation? Oh wait, no: ``a10n``. (It\'s actually a form\n    of `numeronym`_.)\n\n    >>> a10n(\'abbreviation\')\n    \'a10n\'\n    >>> a10n(\'internationalization\')\n    \'i18n\'\n    >>> a10n(\'\')\n    \'\'\n\n    .. _numeronym: http://en.wikipedia.org/wiki/Numeronym\n    '
    if len(string) < 3:
        return string
    return '%s%s%s' % (string[0], len(string[1:-1]), string[-1])
ANSI_SEQUENCES = re.compile('\n    \\x1B            # Sequence starts with ESC, i.e. hex 0x1B\n    (?:\n        [@-Z\\\\-_]   # Second byte:\n                    #   all 0x40–0x5F range but CSI char, i.e ASCII @A–Z\\]^_\n    |               # Or\n        \\[          # CSI sequences, starting with [\n        [0-?]*      # Parameter bytes:\n                    #   range 0x30–0x3F, ASCII 0–9:;<=>?\n        [ -/]*      # Intermediate bytes:\n                    #   range 0x20–0x2F, ASCII space and !"#$%&\'()*+,-./\n        [@-~]       # Final byte\n                    #   range 0x40–0x7E, ASCII @A–Z[\\]^_`a–z{|}~\n    )\n', re.VERBOSE)

def strip_ansi(text):
    if False:
        while True:
            i = 10
    "Strips ANSI escape codes from *text*. Useful for the occasional\n    time when a log or redirected output accidentally captures console\n    color codes and the like.\n\n    >>> strip_ansi('\x1b[0m\x1b[1;36mart\x1b[46;34m')\n    'art'\n\n    Supports unicode, str, bytes and bytearray content as input. Returns the\n    same type as the input.\n\n    There's a lot of ANSI art available for testing on `sixteencolors.net`_.\n    This function does not interpret or render ANSI art, but you can do so with\n    `ansi2img`_ or `escapes.js`_.\n\n    .. _sixteencolors.net: http://sixteencolors.net\n    .. _ansi2img: http://www.bedroomlan.org/projects/ansi2img\n    .. _escapes.js: https://github.com/atdt/escapes.js\n    "
    target_type = None
    is_py3 = unicode == builtins.str
    if is_py3 and isinstance(text, (bytes, bytearray)):
        target_type = type(text)
        text = text.decode('utf-8')
    cleaned = ANSI_SEQUENCES.sub('', text)
    if target_type and target_type != type(cleaned):
        cleaned = target_type(cleaned, 'utf-8')
    return cleaned

def asciify(text, ignore=False):
    if False:
        print('Hello World!')
    "Converts a unicode or bytestring, *text*, into a bytestring with\n    just ascii characters. Performs basic deaccenting for all you\n    Europhiles out there.\n\n    Also, a gentle reminder that this is a **utility**, primarily meant\n    for slugification. Whenever possible, make your application work\n    **with** unicode, not against it.\n\n    Args:\n        text (str or unicode): The string to be asciified.\n        ignore (bool): Configures final encoding to ignore remaining\n            unasciified unicode instead of replacing it.\n\n    >>> asciify('Beyoncé') == b'Beyonce'\n    True\n    "
    try:
        try:
            return text.encode('ascii')
        except UnicodeDecodeError:
            text = text.decode('utf-8')
            return text.encode('ascii')
    except UnicodeEncodeError:
        mode = 'replace'
        if ignore:
            mode = 'ignore'
        transd = unicodedata.normalize('NFKD', text.translate(DEACCENT_MAP))
        ret = transd.encode('ascii', mode)
        return ret

def is_ascii(text):
    if False:
        while True:
            i = 10
    "Check if a unicode or bytestring, *text*, is composed of ascii\n    characters only. Raises :exc:`ValueError` if argument is not text.\n\n    Args:\n        text (str or unicode): The string to be checked.\n\n    >>> is_ascii('Beyoncé')\n    False\n    >>> is_ascii('Beyonce')\n    True\n    "
    if isinstance(text, unicode):
        try:
            text.encode('ascii')
        except UnicodeEncodeError:
            return False
    elif isinstance(text, bytes):
        try:
            text.decode('ascii')
        except UnicodeDecodeError:
            return False
    else:
        raise ValueError('expected text or bytes, not %r' % type(text))
    return True

class DeaccenterDict(dict):
    """A small caching dictionary for deaccenting."""

    def __missing__(self, key):
        if False:
            for i in range(10):
                print('nop')
        ch = self.get(key)
        if ch is not None:
            return ch
        try:
            de = unicodedata.decomposition(unichr(key))
            (p1, _, p2) = de.rpartition(' ')
            if int(p2, 16) == 776:
                ch = self.get(key)
            else:
                ch = int(p1, 16)
        except (IndexError, ValueError):
            ch = self.get(key, key)
        self[key] = ch
        return ch
    try:
        from collections import defaultdict
    except ImportError:

        def __getitem__(self, key):
            if False:
                for i in range(10):
                    print('nop')
            try:
                return super(DeaccenterDict, self).__getitem__(key)
            except KeyError:
                return self.__missing__(key)
    else:
        del defaultdict
_BASE_DEACCENT_MAP = {198: u'AE', 208: u'D', 216: u'OE', 222: u'Th', 196: u'Ae', 214: u'Oe', 220: u'Ue', 192: u'A', 193: u'A', 195: u'A', 199: u'C', 200: u'E', 201: u'E', 202: u'E', 204: u'I', 205: u'I', 210: u'O', 211: u'O', 213: u'O', 217: u'U', 218: u'U', 223: u'ss', 230: u'ae', 240: u'd', 248: u'oe', 254: u'th', 228: u'ae', 246: u'oe', 252: u'ue', 224: u'a', 225: u'a', 227: u'a', 231: u'c', 232: u'e', 233: u'e', 234: u'e', 236: u'i', 237: u'i', 242: u'o', 243: u'o', 245: u'o', 249: u'u', 250: u'u', 8216: u"'", 8217: u"'", 8220: u'"', 8221: u'"'}
DEACCENT_MAP = DeaccenterDict(_BASE_DEACCENT_MAP)
_SIZE_SYMBOLS = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
_SIZE_BOUNDS = [(1024 ** i, sym) for (i, sym) in enumerate(_SIZE_SYMBOLS)]
_SIZE_RANGES = list(zip(_SIZE_BOUNDS, _SIZE_BOUNDS[1:]))

def bytes2human(nbytes, ndigits=0):
    if False:
        while True:
            i = 10
    "Turns an integer value of *nbytes* into a human readable format. Set\n    *ndigits* to control how many digits after the decimal point\n    should be shown (default ``0``).\n\n    >>> bytes2human(128991)\n    '126K'\n    >>> bytes2human(100001221)\n    '95M'\n    >>> bytes2human(0, 2)\n    '0.00B'\n    "
    abs_bytes = abs(nbytes)
    for ((size, symbol), (next_size, next_symbol)) in _SIZE_RANGES:
        if abs_bytes <= next_size:
            break
    hnbytes = float(nbytes) / size
    return '{hnbytes:.{ndigits}f}{symbol}'.format(hnbytes=hnbytes, ndigits=ndigits, symbol=symbol)

class HTMLTextExtractor(HTMLParser):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.result = []

    def handle_data(self, d):
        if False:
            print('Hello World!')
        self.result.append(d)

    def handle_charref(self, number):
        if False:
            i = 10
            return i + 15
        if number[0] == u'x' or number[0] == u'X':
            codepoint = int(number[1:], 16)
        else:
            codepoint = int(number)
        self.result.append(unichr(codepoint))

    def handle_entityref(self, name):
        if False:
            for i in range(10):
                print('nop')
        try:
            codepoint = htmlentitydefs.name2codepoint[name]
        except KeyError:
            self.result.append(u'&' + name + u';')
        else:
            self.result.append(unichr(codepoint))

    def get_text(self):
        if False:
            for i in range(10):
                print('nop')
        return u''.join(self.result)

def html2text(html):
    if False:
        i = 10
        return i + 15
    'Strips tags from HTML text, returning markup-free text. Also, does\n    a best effort replacement of entities like "&nbsp;"\n\n    >>> r = html2text(u\'<a href="#">Test &amp;<em>(Δ&#x03b7;&#956;&#x03CE;)</em></a>\')\n    >>> r == u\'Test &(Δημώ)\'\n    True\n    '
    s = HTMLTextExtractor()
    s.feed(html)
    return s.get_text()
_EMPTY_GZIP_BYTES = b'\x1f\x8b\x08\x089\xf3\xb9U\x00\x03empty\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00'
_NON_EMPTY_GZIP_BYTES = b'\x1f\x8b\x08\x08\xbc\xf7\xb9U\x00\x03not_empty\x00K\xaa,I-N\xcc\xc8\xafT\xe4\x02\x00\xf3nb\xbf\x0b\x00\x00\x00'

def gunzip_bytes(bytestring):
    if False:
        for i in range(10):
            print('nop')
    "The :mod:`gzip` module is great if you have a file or file-like\n    object, but what if you just have bytes. StringIO is one\n    possibility, but it's often faster, easier, and simpler to just\n    use this one-liner. Use this tried-and-true utility function to\n    decompress gzip from bytes.\n\n    >>> gunzip_bytes(_EMPTY_GZIP_BYTES) == b''\n    True\n    >>> gunzip_bytes(_NON_EMPTY_GZIP_BYTES).rstrip() == b'bytesahoy!'\n    True\n    "
    return zlib.decompress(bytestring, 16 + zlib.MAX_WBITS)

def gzip_bytes(bytestring, level=6):
    if False:
        while True:
            i = 10
    "Turn some bytes into some compressed bytes.\n\n    >>> len(gzip_bytes(b'a' * 10000))\n    46\n\n    Args:\n        bytestring (bytes): Bytes to be compressed\n        level (int): An integer, 1-9, controlling the\n          speed/compression. 1 is fastest, least compressed, 9 is\n          slowest, but most compressed.\n\n    Note that all levels of gzip are pretty fast these days, though\n    it's not really a competitor in compression, at any level.\n    "
    out = StringIO()
    f = GzipFile(fileobj=out, mode='wb', compresslevel=level)
    f.write(bytestring)
    f.close()
    return out.getvalue()
_line_ending_re = re.compile('(\\r\\n|\\n|\\x0b|\\f|\\r|\\x85|\\x2028|\\x2029)', re.UNICODE)

def iter_splitlines(text):
    if False:
        print('Hello World!')
    "Like :meth:`str.splitlines`, but returns an iterator of lines\n    instead of a list. Also similar to :meth:`file.next`, as that also\n    lazily reads and yields lines from a file.\n\n    This function works with a variety of line endings, but as always,\n    be careful when mixing line endings within a file.\n\n    >>> list(iter_splitlines('\\nhi\\nbye\\n'))\n    ['', 'hi', 'bye', '']\n    >>> list(iter_splitlines('\\r\\nhi\\rbye\\r\\n'))\n    ['', 'hi', 'bye', '']\n    >>> list(iter_splitlines(''))\n    []\n    "
    (prev_end, len_text) = (0, len(text))
    for match in _line_ending_re.finditer(text):
        (start, end) = (match.start(1), match.end(1))
        if prev_end <= start:
            yield text[prev_end:start]
        if end == len_text:
            yield ''
        prev_end = end
    tail = text[prev_end:]
    if tail:
        yield tail
    return

def indent(text, margin, newline='\n', key=bool):
    if False:
        print('Hello World!')
    'The missing counterpart to the built-in :func:`textwrap.dedent`.\n\n    Args:\n        text (str): The text to indent.\n        margin (str): The string to prepend to each line.\n        newline (str): The newline used to rejoin the lines (default: ``\\n``)\n        key (callable): Called on each line to determine whether to\n          indent it. Default: :class:`bool`, to ensure that empty lines do\n          not get whitespace added.\n    '
    indented_lines = [margin + line if key(line) else line for line in iter_splitlines(text)]
    return newline.join(indented_lines)

def is_uuid(obj, version=4):
    if False:
        i = 10
        return i + 15
    "Check the argument is either a valid UUID object or string.\n\n    Args:\n        obj (object): The test target. Strings and UUID objects supported.\n        version (int): The target UUID version, set to 0 to skip version check.\n\n    >>> is_uuid('e682ccca-5a4c-4ef2-9711-73f9ad1e15ea')\n    True\n    >>> is_uuid('0221f0d9-d4b9-11e5-a478-10ddb1c2feb9')\n    False\n    >>> is_uuid('0221f0d9-d4b9-11e5-a478-10ddb1c2feb9', version=1)\n    True\n    "
    if not isinstance(obj, uuid.UUID):
        try:
            obj = uuid.UUID(obj)
        except (TypeError, ValueError, AttributeError):
            return False
    if version and obj.version != int(version):
        return False
    return True

def escape_shell_args(args, sep=' ', style=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns an escaped version of each string in *args*, according to\n    *style*.\n\n    Args:\n        args (list): A list of arguments to escape and join together\n        sep (str): The separator used to join the escaped arguments.\n        style (str): The style of escaping to use. Can be one of\n          ``cmd`` or ``sh``, geared toward Windows and Linux/BSD/etc.,\n          respectively. If *style* is ``None``, then it is picked\n          according to the system platform.\n\n    See :func:`args2cmd` and :func:`args2sh` for details and example\n    output for each style.\n    '
    if not style:
        style = 'cmd' if sys.platform == 'win32' else 'sh'
    if style == 'sh':
        return args2sh(args, sep=sep)
    elif style == 'cmd':
        return args2cmd(args, sep=sep)
    raise ValueError("style expected one of 'cmd' or 'sh', not %r" % style)
_find_sh_unsafe = re.compile('[^a-zA-Z0-9_@%+=:,./-]').search

def args2sh(args, sep=' '):
    if False:
        for i in range(10):
            print('nop')
    'Return a shell-escaped string version of *args*, separated by\n    *sep*, based on the rules of sh, bash, and other shells in the\n    Linux/BSD/MacOS ecosystem.\n\n    >>> print(args2sh([\'aa\', \'[bb]\', "cc\'cc", \'dd"dd\']))\n    aa \'[bb]\' \'cc\'"\'"\'cc\' \'dd"dd\'\n\n    As you can see, arguments with no special characters are not\n    escaped, arguments with special characters are quoted with single\n    quotes, and single quotes themselves are quoted with double\n    quotes. Double quotes are handled like any other special\n    character.\n\n    Based on code from the :mod:`pipes`/:mod:`shlex` modules. Also\n    note that :mod:`shlex` and :mod:`argparse` have functions to split\n    and parse strings escaped in this manner.\n    '
    ret_list = []
    for arg in args:
        if not arg:
            ret_list.append("''")
            continue
        if _find_sh_unsafe(arg) is None:
            ret_list.append(arg)
            continue
        ret_list.append("'" + arg.replace("'", '\'"\'"\'') + "'")
    return ' '.join(ret_list)

def args2cmd(args, sep=' '):
    if False:
        while True:
            i = 10
    'Return a shell-escaped string version of *args*, separated by\n    *sep*, using the same rules as the Microsoft C runtime.\n\n    >>> print(args2cmd([\'aa\', \'[bb]\', "cc\'cc", \'dd"dd\']))\n    aa [bb] cc\'cc dd\\"dd\n\n    As you can see, escaping is through backslashing and not quoting,\n    and double quotes are the only special character. See the comment\n    in the code for more details. Based on internal code from the\n    :mod:`subprocess` module.\n\n    '
    '\n    1) Arguments are delimited by white space, which is either a\n       space or a tab.\n\n    2) A string surrounded by double quotation marks is\n       interpreted as a single argument, regardless of white space\n       contained within.  A quoted string can be embedded in an\n       argument.\n\n    3) A double quotation mark preceded by a backslash is\n       interpreted as a literal double quotation mark.\n\n    4) Backslashes are interpreted literally, unless they\n       immediately precede a double quotation mark.\n\n    5) If backslashes immediately precede a double quotation mark,\n       every pair of backslashes is interpreted as a literal\n       backslash.  If the number of backslashes is odd, the last\n       backslash escapes the next double quotation mark as\n       described in rule 3.\n\n    See http://msdn.microsoft.com/en-us/library/17w5ykft.aspx\n    or search http://msdn.microsoft.com for\n    "Parsing C++ Command-Line Arguments"\n    '
    result = []
    needquote = False
    for arg in args:
        bs_buf = []
        if result:
            result.append(' ')
        needquote = ' ' in arg or '\t' in arg or (not arg)
        if needquote:
            result.append('"')
        for c in arg:
            if c == '\\':
                bs_buf.append(c)
            elif c == '"':
                result.append('\\' * len(bs_buf) * 2)
                bs_buf = []
                result.append('\\"')
            else:
                if bs_buf:
                    result.extend(bs_buf)
                    bs_buf = []
                result.append(c)
        if bs_buf:
            result.extend(bs_buf)
        if needquote:
            result.extend(bs_buf)
            result.append('"')
    return ''.join(result)

def parse_int_list(range_string, delim=',', range_delim='-'):
    if False:
        for i in range(10):
            print('nop')
    "Returns a sorted list of positive integers based on\n    *range_string*. Reverse of :func:`format_int_list`.\n\n    Args:\n        range_string (str): String of comma separated positive\n            integers or ranges (e.g. '1,2,4-6,8'). Typical of a custom\n            page range string used in printer dialogs.\n        delim (char): Defaults to ','. Separates integers and\n            contiguous ranges of integers.\n        range_delim (char): Defaults to '-'. Indicates a contiguous\n            range of integers.\n\n    >>> parse_int_list('1,3,5-8,10-11,15')\n    [1, 3, 5, 6, 7, 8, 10, 11, 15]\n\n    "
    output = []
    for x in range_string.strip().split(delim):
        if range_delim in x:
            range_limits = list(map(int, x.split(range_delim)))
            output += list(range(min(range_limits), max(range_limits) + 1))
        elif not x:
            continue
        else:
            output.append(int(x))
    return sorted(output)

def format_int_list(int_list, delim=',', range_delim='-', delim_space=False):
    if False:
        for i in range(10):
            print('nop')
    "Returns a sorted range string from a list of positive integers\n    (*int_list*). Contiguous ranges of integers are collapsed to min\n    and max values. Reverse of :func:`parse_int_list`.\n\n    Args:\n        int_list (list): List of positive integers to be converted\n           into a range string (e.g. [1,2,4,5,6,8]).\n        delim (char): Defaults to ','. Separates integers and\n           contiguous ranges of integers.\n        range_delim (char): Defaults to '-'. Indicates a contiguous\n           range of integers.\n        delim_space (bool): Defaults to ``False``. If ``True``, adds a\n           space after all *delim* characters.\n\n    >>> format_int_list([1,3,5,6,7,8,10,11,15])\n    '1,3,5-8,10-11,15'\n\n    "
    output = []
    contig_range = collections.deque()
    for x in sorted(int_list):
        if len(contig_range) < 1:
            contig_range.append(x)
        elif len(contig_range) > 1:
            delta = x - contig_range[-1]
            if delta == 1:
                contig_range.append(x)
            elif delta > 1:
                range_substr = '{0:d}{1}{2:d}'.format(min(contig_range), range_delim, max(contig_range))
                output.append(range_substr)
                contig_range.clear()
                contig_range.append(x)
            else:
                continue
        else:
            delta = x - contig_range[0]
            if delta == 1:
                contig_range.append(x)
            elif delta > 1:
                output.append('{0:d}'.format(contig_range.popleft()))
                contig_range.append(x)
            else:
                continue
    else:
        if len(contig_range) == 1:
            output.append('{0:d}'.format(contig_range.popleft()))
            contig_range.clear()
        elif len(contig_range) > 1:
            range_substr = '{0:d}{1}{2:d}'.format(min(contig_range), range_delim, max(contig_range))
            output.append(range_substr)
            contig_range.clear()
    if delim_space:
        output_str = (delim + ' ').join(output)
    else:
        output_str = delim.join(output)
    return output_str

def complement_int_list(range_string, range_start=0, range_end=None, delim=',', range_delim='-'):
    if False:
        return 10
    " Returns range string that is the complement of the one provided as\n    *range_string* parameter.\n\n    These range strings are of the kind produce by :func:`format_int_list`, and\n    parseable by :func:`parse_int_list`.\n\n    Args:\n        range_string (str): String of comma separated positive integers or\n           ranges (e.g. '1,2,4-6,8'). Typical of a custom page range string\n           used in printer dialogs.\n        range_start (int): A positive integer from which to start the resulting\n           range. Value is inclusive. Defaults to ``0``.\n        range_end (int): A positive integer from which the produced range is\n           stopped. Value is exclusive. Defaults to the maximum value found in\n           the provided ``range_string``.\n        delim (char): Defaults to ','. Separates integers and contiguous ranges\n           of integers.\n        range_delim (char): Defaults to '-'. Indicates a contiguous range of\n           integers.\n\n    >>> complement_int_list('1,3,5-8,10-11,15')\n    '0,2,4,9,12-14'\n\n    >>> complement_int_list('1,3,5-8,10-11,15', range_start=0)\n    '0,2,4,9,12-14'\n\n    >>> complement_int_list('1,3,5-8,10-11,15', range_start=1)\n    '2,4,9,12-14'\n\n    >>> complement_int_list('1,3,5-8,10-11,15', range_start=2)\n    '2,4,9,12-14'\n\n    >>> complement_int_list('1,3,5-8,10-11,15', range_start=3)\n    '4,9,12-14'\n\n    >>> complement_int_list('1,3,5-8,10-11,15', range_end=15)\n    '0,2,4,9,12-14'\n\n    >>> complement_int_list('1,3,5-8,10-11,15', range_end=14)\n    '0,2,4,9,12-13'\n\n    >>> complement_int_list('1,3,5-8,10-11,15', range_end=13)\n    '0,2,4,9,12'\n\n    >>> complement_int_list('1,3,5-8,10-11,15', range_end=20)\n    '0,2,4,9,12-14,16-19'\n\n    >>> complement_int_list('1,3,5-8,10-11,15', range_end=0)\n    ''\n\n    >>> complement_int_list('1,3,5-8,10-11,15', range_start=-1)\n    '0,2,4,9,12-14'\n\n    >>> complement_int_list('1,3,5-8,10-11,15', range_end=-1)\n    ''\n\n    >>> complement_int_list('1,3,5-8', range_start=1, range_end=1)\n    ''\n\n    >>> complement_int_list('1,3,5-8', range_start=2, range_end=2)\n    ''\n\n    >>> complement_int_list('1,3,5-8', range_start=2, range_end=3)\n    '2'\n\n    >>> complement_int_list('1,3,5-8', range_start=-10, range_end=-5)\n    ''\n\n    >>> complement_int_list('1,3,5-8', range_start=20, range_end=10)\n    ''\n\n    >>> complement_int_list('')\n    ''\n    "
    int_list = set(parse_int_list(range_string, delim, range_delim))
    if range_end is None:
        if int_list:
            range_end = max(int_list) + 1
        else:
            range_end = range_start
    complement_values = set(range(range_end)) - int_list - set(range(range_start))
    return format_int_list(complement_values, delim, range_delim)

def int_ranges_from_int_list(range_string, delim=',', range_delim='-'):
    if False:
        for i in range(10):
            print('nop')
    " Transform a string of ranges (*range_string*) into a tuple of tuples.\n\n    Args:\n        range_string (str): String of comma separated positive integers or\n           ranges (e.g. '1,2,4-6,8'). Typical of a custom page range string\n           used in printer dialogs.\n        delim (char): Defaults to ','. Separates integers and contiguous ranges\n           of integers.\n        range_delim (char): Defaults to '-'. Indicates a contiguous range of\n           integers.\n\n    >>> int_ranges_from_int_list('1,3,5-8,10-11,15')\n    ((1, 1), (3, 3), (5, 8), (10, 11), (15, 15))\n\n    >>> int_ranges_from_int_list('1')\n    ((1, 1),)\n\n    >>> int_ranges_from_int_list('')\n    ()\n    "
    int_tuples = []
    range_string = format_int_list(parse_int_list(range_string, delim, range_delim))
    if range_string:
        for bounds in range_string.split(','):
            if '-' in bounds:
                (start, end) = bounds.split('-')
            else:
                (start, end) = (bounds, bounds)
            int_tuples.append((int(start), int(end)))
    return tuple(int_tuples)

class MultiReplace(object):
    """
    MultiReplace is a tool for doing multiple find/replace actions in one pass.

    Given a mapping of values to be replaced it allows for all of the matching
    values to be replaced in a single pass which can save a lot of performance
    on very large strings. In addition to simple replace, it also allows for
    replacing based on regular expressions.

    Keyword Arguments:

    :type regex: bool
    :param regex: Treat search keys as regular expressions [Default: False]
    :type flags: int
    :param flags: flags to pass to the regex engine during compile

    Dictionary Usage::

        from boltons import stringutils
        s = stringutils.MultiReplace({
            'foo': 'zoo',
            'cat': 'hat',
            'bat': 'kraken'
        })
        new = s.sub('The foo bar cat ate a bat')
        new == 'The zoo bar hat ate a kraken'

    Iterable Usage::

        from boltons import stringutils
        s = stringutils.MultiReplace([
            ('foo', 'zoo'),
            ('cat', 'hat'),
            ('bat', 'kraken)'
        ])
        new = s.sub('The foo bar cat ate a bat')
        new == 'The zoo bar hat ate a kraken'


    The constructor can be passed a dictionary or other mapping as well as
    an iterable of tuples. If given an iterable, the substitution will be run
    in the order the replacement values are specified in the iterable. This is
    also true if it is given an OrderedDict. If given a dictionary then the
    order will be non-deterministic::

        >>> 'foo bar baz'.replace('foo', 'baz').replace('baz', 'bar')
        'bar bar bar'
        >>> m = MultiReplace({'foo': 'baz', 'baz': 'bar'})
        >>> m.sub('foo bar baz')
        'baz bar bar'

    This is because the order of replacement can matter if you're inserting
    something that might be replaced by a later substitution. Pay attention and
    if you need to rely on order then consider using a list of tuples instead
    of a dictionary.
    """

    def __init__(self, sub_map, **kwargs):
        if False:
            while True:
                i = 10
        'Compile any regular expressions that have been passed.'
        options = {'regex': False, 'flags': 0}
        options.update(kwargs)
        self.group_map = {}
        regex_values = []
        if isinstance(sub_map, Mapping):
            sub_map = sub_map.items()
        for (idx, vals) in enumerate(sub_map):
            group_name = 'group{0}'.format(idx)
            if isinstance(vals[0], basestring):
                if not options['regex']:
                    exp = re.escape(vals[0])
                else:
                    exp = vals[0]
            else:
                exp = vals[0].pattern
            regex_values.append('(?P<{}>{})'.format(group_name, exp))
            self.group_map[group_name] = vals[1]
        self.combined_pattern = re.compile('|'.join(regex_values), flags=options['flags'])

    def _get_value(self, match):
        if False:
            print('Hello World!')
        'Given a match object find replacement value.'
        group_dict = match.groupdict()
        key = [x for x in group_dict if group_dict[x]][0]
        return self.group_map[key]

    def sub(self, text):
        if False:
            i = 10
            return i + 15
        '\n        Run substitutions on the input text.\n\n        Given an input string, run all substitutions given in the\n        constructor.\n        '
        return self.combined_pattern.sub(self._get_value, text)

def multi_replace(text, sub_map, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Shortcut function to invoke MultiReplace in a single call.\n\n    Example Usage::\n\n        from boltons.stringutils import multi_replace\n        new = multi_replace(\n            'The foo bar cat ate a bat',\n            {'foo': 'zoo', 'cat': 'hat', 'bat': 'kraken'}\n        )\n        new == 'The zoo bar hat ate a kraken'\n    "
    m = MultiReplace(sub_map, **kwargs)
    return m.sub(text)

def unwrap_text(text, ending='\n\n'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Unwrap text, the natural complement to :func:`textwrap.wrap`.\n\n    >>> text = "Short \\n lines  \\nwrapped\\nsmall.\\n\\nAnother\\nparagraph."\n    >>> unwrap_text(text)\n    \'Short lines wrapped small.\\n\\nAnother paragraph.\'\n\n    Args:\n       text: A string to unwrap.\n       ending (str): The string to join all unwrapped paragraphs\n          by. Pass ``None`` to get the list. Defaults to \'\\n\\n\' for\n          compatibility with Markdown and RST.\n\n    '
    all_grafs = []
    cur_graf = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            cur_graf.append(line)
        else:
            all_grafs.append(' '.join(cur_graf))
            cur_graf = []
    if cur_graf:
        all_grafs.append(' '.join(cur_graf))
    if ending is None:
        return all_grafs
    return ending.join(all_grafs)