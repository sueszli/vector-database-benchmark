"""
Twitter-aware tokenizer, designed to be flexible and easy to adapt to new
domains and tasks. The basic logic is this:

1. The tuple REGEXPS defines a list of regular expression
   strings.

2. The REGEXPS strings are put, in order, into a compiled
   regular expression object called WORD_RE, under the TweetTokenizer
   class.

3. The tokenization is done by WORD_RE.findall(s), where s is the
   user-supplied string, inside the tokenize() method of the class
   TweetTokenizer.

4. When instantiating Tokenizer objects, there are several options:
    * preserve_case. By default, it is set to True. If it is set to
      False, then the tokenizer will downcase everything except for
      emoticons.
    * reduce_len. By default, it is set to False. It specifies whether
      to replace repeated character sequences of length 3 or greater
      with sequences of length 3.
    * strip_handles. By default, it is set to False. It specifies
      whether to remove Twitter handles of text used in the
      `tokenize` method.
    * match_phone_numbers. By default, it is set to True. It indicates
      whether the `tokenize` method should look for phone numbers.
"""
import html
from typing import List
import regex
from nltk.tokenize.api import TokenizerI
EMOTICONS = "\n    (?:\n      [<>]?\n      [:;=8]                     # eyes\n      [\\-o\\*\\']?                 # optional nose\n      [\\)\\]\\(\\[dDpP/\\:\\}\\{@\\|\\\\] # mouth\n      |\n      [\\)\\]\\(\\[dDpP/\\:\\}\\{@\\|\\\\] # mouth\n      [\\-o\\*\\']?                 # optional nose\n      [:;=8]                     # eyes\n      [<>]?\n      |\n      </?3                       # heart\n    )"
URLS = '\t\t\t# Capture 1: entire matched URL\n  (?:\n  https?:\t\t\t\t# URL protocol and colon\n    (?:\n      /{1,3}\t\t\t\t# 1-3 slashes\n      |\t\t\t\t\t#   or\n      [a-z0-9%]\t\t\t\t# Single letter or digit or \'%\'\n                                       # (Trying not to match e.g. "URI::Escape")\n    )\n    |\t\t\t\t\t#   or\n                                       # looks like domain name followed by a slash:\n    [a-z0-9.\\-]+[.]\n    (?:[a-z]{2,13})\n    /\n  )\n  (?:\t\t\t\t\t# One or more:\n    [^\\s()<>{}\\[\\]]+\t\t\t# Run of non-space, non-()<>{}[]\n    |\t\t\t\t\t#   or\n    \\([^\\s()]*?\\([^\\s()]+\\)[^\\s()]*?\\) # balanced parens, one level deep: (...(...)...)\n    |\n    \\([^\\s]+?\\)\t\t\t\t# balanced parens, non-recursive: (...)\n  )+\n  (?:\t\t\t\t\t# End with:\n    \\([^\\s()]*?\\([^\\s()]+\\)[^\\s()]*?\\) # balanced parens, one level deep: (...(...)...)\n    |\n    \\([^\\s]+?\\)\t\t\t\t# balanced parens, non-recursive: (...)\n    |\t\t\t\t\t#   or\n    [^\\s`!()\\[\\]{};:\'".,<>?«»“”‘’]\t# not a space or one of these punct chars\n  )\n  |\t\t\t\t\t# OR, the following to match naked domains:\n  (?:\n  \t(?<!@)\t\t\t        # not preceded by a @, avoid matching foo@_gmail.com_\n    [a-z0-9]+\n    (?:[.\\-][a-z0-9]+)*\n    [.]\n    (?:[a-z]{2,13})\n    \\b\n    /?\n    (?!@)\t\t\t        # not succeeded by a @,\n                            # avoid matching "foo.na" in "foo.na@example.com"\n  )\n'
FLAGS = '\n  (?:\n    [\\U0001F1E6-\\U0001F1FF]{2}  # all enclosed letter pairs\n    |\n    # English flag\n    \\U0001F3F4\\U000E0067\\U000E0062\\U000E0065\\U000E006e\\U000E0067\\U000E007F\n    |\n    # Scottish flag\n    \\U0001F3F4\\U000E0067\\U000E0062\\U000E0073\\U000E0063\\U000E0074\\U000E007F\n    |\n    # For Wales? Why Richard, it profit a man nothing to give his soul for the whole world … but for Wales!\n    \\U0001F3F4\\U000E0067\\U000E0062\\U000E0077\\U000E006C\\U000E0073\\U000E007F\n  )\n'
PHONE_REGEX = '\n    (?:\n      (?:            # (international)\n        \\+?[01]\n        [ *\\-.\\)]*\n      )?\n      (?:            # (area code)\n        [\\(]?\n        \\d{3}\n        [ *\\-.\\)]*\n      )?\n      \\d{3}          # exchange\n      [ *\\-.\\)]*\n      \\d{4}          # base\n    )'
REGEXPS = (URLS, EMOTICONS, '<[^>\\s]+>', '[\\-]+>|<[\\-]+', '(?:@[\\w_]+)', "(?:\\#+[\\w_]+[\\w\\'_\\-]*[\\w_]+)", '[\\w.+-]+@[\\w-]+\\.(?:[\\w-]\\.?)+[\\w-]', '.(?:\n        [🏻-🏿]?(?:\u200d.[🏻-🏿]?)+\n        |\n        [🏻-🏿]\n    )', FLAGS, "\n    (?:[^\\W\\d_](?:[^\\W\\d_]|['\\-_])+[^\\W\\d_]) # Words with apostrophes or dashes.\n    |\n    (?:[+\\-]?\\d+[,/.:-]\\d+[+\\-]?)  # Numbers, including fractions, decimals.\n    |\n    (?:[\\w_]+)                     # Words without apostrophes or dashes.\n    |\n    (?:\\.(?:\\s*\\.){1,})            # Ellipsis dots.\n    |\n    (?:\\S)                         # Everything else that isn't whitespace.\n    ")
REGEXPS_PHONE = (REGEXPS[0], PHONE_REGEX, *REGEXPS[1:])
HANG_RE = regex.compile('([^a-zA-Z0-9])\\1{3,}')
EMOTICON_RE = regex.compile(EMOTICONS, regex.VERBOSE | regex.I | regex.UNICODE)
ENT_RE = regex.compile('&(#?(x?))([^&;\\s]+);')
HANDLES_RE = regex.compile('(?<![A-Za-z0-9_!@#\\$%&*])@(([A-Za-z0-9_]){15}(?!@)|([A-Za-z0-9_]){1,14}(?![A-Za-z0-9_]*@))')

def _str_to_unicode(text, encoding=None, errors='strict'):
    if False:
        print('Hello World!')
    if encoding is None:
        encoding = 'utf-8'
    if isinstance(text, bytes):
        return text.decode(encoding, errors)
    return text

def _replace_html_entities(text, keep=(), remove_illegal=True, encoding='utf-8'):
    if False:
        while True:
            i = 10
    '\n    Remove entities from text by converting them to their\n    corresponding unicode character.\n\n    :param text: a unicode string or a byte string encoded in the given\n    `encoding` (which defaults to \'utf-8\').\n\n    :param list keep:  list of entity names which should not be replaced.    This supports both numeric entities (``&#nnnn;`` and ``&#hhhh;``)\n    and named entities (such as ``&nbsp;`` or ``&gt;``).\n\n    :param bool remove_illegal: If `True`, entities that can\'t be converted are    removed. Otherwise, entities that can\'t be converted are kept "as\n    is".\n\n    :returns: A unicode string with the entities removed.\n\n    See https://github.com/scrapy/w3lib/blob/master/w3lib/html.py\n\n        >>> from nltk.tokenize.casual import _replace_html_entities\n        >>> _replace_html_entities(b\'Price: &pound;100\')\n        \'Price: \\xa3100\'\n        >>> print(_replace_html_entities(b\'Price: &pound;100\'))\n        Price: £100\n        >>>\n    '

    def _convert_entity(match):
        if False:
            for i in range(10):
                print('nop')
        entity_body = match.group(3)
        if match.group(1):
            try:
                if match.group(2):
                    number = int(entity_body, 16)
                else:
                    number = int(entity_body, 10)
                if 128 <= number <= 159:
                    return bytes((number,)).decode('cp1252')
            except ValueError:
                number = None
        else:
            if entity_body in keep:
                return match.group(0)
            number = html.entities.name2codepoint.get(entity_body)
        if number is not None:
            try:
                return chr(number)
            except (ValueError, OverflowError):
                pass
        return '' if remove_illegal else match.group(0)
    return ENT_RE.sub(_convert_entity, _str_to_unicode(text, encoding))

class TweetTokenizer(TokenizerI):
    """
    Tokenizer for tweets.

        >>> from nltk.tokenize import TweetTokenizer
        >>> tknzr = TweetTokenizer()
        >>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
        >>> tknzr.tokenize(s0) # doctest: +NORMALIZE_WHITESPACE
        ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->',
         '<--']

    Examples using `strip_handles` and `reduce_len parameters`:

        >>> tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        >>> s1 = '@remy: This is waaaaayyyy too much for you!!!!!!'
        >>> tknzr.tokenize(s1)
        [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']
    """
    _WORD_RE = None
    _PHONE_WORD_RE = None

    def __init__(self, preserve_case=True, reduce_len=False, strip_handles=False, match_phone_numbers=True):
        if False:
            while True:
                i = 10
        '\n        Create a `TweetTokenizer` instance with settings for use in the `tokenize` method.\n\n        :param preserve_case: Flag indicating whether to preserve the casing (capitalisation)\n            of text used in the `tokenize` method. Defaults to True.\n        :type preserve_case: bool\n        :param reduce_len: Flag indicating whether to replace repeated character sequences\n            of length 3 or greater with sequences of length 3. Defaults to False.\n        :type reduce_len: bool\n        :param strip_handles: Flag indicating whether to remove Twitter handles of text used\n            in the `tokenize` method. Defaults to False.\n        :type strip_handles: bool\n        :param match_phone_numbers: Flag indicating whether the `tokenize` method should look\n            for phone numbers. Defaults to True.\n        :type match_phone_numbers: bool\n        '
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles
        self.match_phone_numbers = match_phone_numbers

    def tokenize(self, text: str) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Tokenize the input text.\n\n        :param text: str\n        :rtype: list(str)\n        :return: a tokenized list of strings; joining this list returns        the original string if `preserve_case=False`.\n        '
        text = _replace_html_entities(text)
        if self.strip_handles:
            text = remove_handles(text)
        if self.reduce_len:
            text = reduce_lengthening(text)
        safe_text = HANG_RE.sub('\\1\\1\\1', text)
        if self.match_phone_numbers:
            words = self.PHONE_WORD_RE.findall(safe_text)
        else:
            words = self.WORD_RE.findall(safe_text)
        if not self.preserve_case:
            words = list(map(lambda x: x if EMOTICON_RE.search(x) else x.lower(), words))
        return words

    @property
    def WORD_RE(self) -> 'regex.Pattern':
        if False:
            print('Hello World!')
        'Core TweetTokenizer regex'
        if not type(self)._WORD_RE:
            type(self)._WORD_RE = regex.compile(f"({'|'.join(REGEXPS)})", regex.VERBOSE | regex.I | regex.UNICODE)
        return type(self)._WORD_RE

    @property
    def PHONE_WORD_RE(self) -> 'regex.Pattern':
        if False:
            return 10
        'Secondary core TweetTokenizer regex'
        if not type(self)._PHONE_WORD_RE:
            type(self)._PHONE_WORD_RE = regex.compile(f"({'|'.join(REGEXPS_PHONE)})", regex.VERBOSE | regex.I | regex.UNICODE)
        return type(self)._PHONE_WORD_RE

def reduce_lengthening(text):
    if False:
        while True:
            i = 10
    '\n    Replace repeated character sequences of length 3 or greater with sequences\n    of length 3.\n    '
    pattern = regex.compile('(.)\\1{2,}')
    return pattern.sub('\\1\\1\\1', text)

def remove_handles(text):
    if False:
        return 10
    '\n    Remove Twitter username handles from text.\n    '
    return HANDLES_RE.sub(' ', text)

def casual_tokenize(text, preserve_case=True, reduce_len=False, strip_handles=False, match_phone_numbers=True):
    if False:
        print('Hello World!')
    '\n    Convenience function for wrapping the tokenizer.\n    '
    return TweetTokenizer(preserve_case=preserve_case, reduce_len=reduce_len, strip_handles=strip_handles, match_phone_numbers=match_phone_numbers).tokenize(text)